"""Experiments E1-E4 and the ablation. Config in, results JSON out; no hardcoded results."""
from __future__ import annotations

import json
import subprocess
import time
import traceback
from pathlib import Path

import numpy as np
import torch

from .alloc import allocate, group_sensitivity, variance_sensitivity
from .channel import Plan, build_plan, transmit
from .complexity import count_macs, count_params
from .config import config_hash
from .features import logmel_to_wav, mel_filterbank, wav_to_logmel
from .metrics import accuracy_per_clip, mel_l1_per_clip, stoi_per_clip
from .quant import kbps, quantize, total_bits
from .stats import bootstrap_ci, paired_bootstrap
from .tasks import predict
from .train import encode_all, latent_stats

TASKS = ("kws", "sid")


class Codec:
    """One trained autoencoder plus everything derived from it (latents, stats, sensitivities)."""

    def __init__(self, ae, bundle, cfg: dict, groups: int, seed: int):
        self.ae, self.b, self.cfg, self.G, self.seed = ae, bundle, cfg, groups, seed
        ev = cfg["eval"]["batch_size"]
        n_stats = min(len(bundle.ae_train), cfg["ae"]["stats_clips"])
        self.mu, self.sigma = latent_stats(encode_all(ae, bundle.ae_train[:n_stats], ev))
        self.z = {(t, s): encode_all(ae, getattr(bundle, f"{t}_{s}").x, ev)
                  for t in TASKS for s in ("val", "test")}
        self.T = self.z[("kws", "test")].shape[1]
        self.d = ae.d
        self.group_sizes = np.full(groups, self.T * self.d, dtype=np.int64)
        self.n_elems = int(self.group_sizes.sum())
        self.models = {"kws": bundle.kws, "sid": bundle.sid}
        self._sens = None

    # sensitivities (validation only)
    def sens(self) -> dict:
        if self._sens is None:
            s = {"variance": variance_sensitivity(self.sigma)}
            for t in TASKS:
                s[f"task_{t}"] = group_sensitivity(self.ae.decode, self.models[t], self.z[(t, "val")],
                                                   getattr(self.b, f"{t}_val").y, self.sigma,
                                                   self.cfg["sensitivity"]["max_clips"])
            self._sens = s
        return self._sens

    def budget_bits(self, bpe: float) -> int:
        return int(round(bpe * self.n_elems))

    def plan(self, scheme: str, policy: str, budget: int, prot_task: str, seed: int = 0) -> Plan:
        s = self.sens()
        sens_alloc = s.get(policy) if policy not in ("uniform", "random") else None
        lat = self.cfg["latent"]
        return build_plan(scheme, policy, sens_alloc, s[f"task_{prot_task}"], budget, self.group_sizes,
                          lat["b_min"], lat["b_max"], self.cfg["channel"]["protect_frac"], seed)

    def quantized(self, task: str, split: str, bits) -> torch.Tensor:
        return quantize(self.z[(task, split)], torch.as_tensor(bits), self.mu, self.sigma,
                        self.cfg["latent"]["clip_k"], ste=False)

    @torch.no_grad()
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        bs = self.cfg["eval"]["batch_size"]
        return torch.cat([self.ae.decode(z[i:i + bs]) for i in range(0, len(z), bs)])

    @torch.no_grad()
    def correct(self, task: str, plan: Plan | None, loss_cfg: dict | None = None,
                rng_seed: int = 0, split: str = "test", float_latent: bool = False) -> np.ndarray:
        """Per-clip correctness (mean over channel draws) of `task` on reconstructed features."""
        sp = getattr(self.b, f"{task}_{split}")
        if float_latent:
            xh = self.decode(self.z[(task, split)])
            return accuracy_per_clip(predict(self.models[task], xh), sp.y)
        q = self.quantized(task, split, plan.bits)
        lossy = loss_cfg is not None and loss_cfg.get("p", 0.0) > 0
        draws = self.cfg["channel"]["draws"] if lossy else 1
        acc = np.zeros(len(sp.y))
        for k in range(draws):
            if lossy:
                rng = np.random.default_rng([self.seed, rng_seed, k])
                rx = transmit(q, plan, loss_cfg, rng, self.mu, self.cfg["channel"]["chunk_len"])
            else:
                rx = q
            acc += accuracy_per_clip(predict(self.models[task], self.decode(rx)), sp.y)
        return acc / draws

    @torch.no_grad()
    def recon_logmel(self, task: str, plan: Plan, split: str = "test") -> torch.Tensor:
        """Reconstructed UN-normalised log-mel."""
        return self.b.norm.invert(self.decode(self.quantized(task, split, plan.bits)))


def _ci(cfg, x):
    m, lo, hi = bootstrap_ci(x, cfg["eval"]["bootstrap"], cfg["eval"]["ci"], cfg["seed"])
    return {"mean": m, "lo": lo, "hi": hi}


def _pd(cfg, a, b):
    m, lo, hi = paired_bootstrap(a, b, cfg["eval"]["bootstrap"], cfg["eval"]["ci"], cfg["seed"])
    return {"diff": m, "lo": lo, "hi": hi}


def _avg(arrs):
    return np.mean(np.stack(arrs), axis=0)


def eval_policy(codec: Codec, cfg: dict, task: str, policy: str, bpe: float, scheme: str = "none",
                loss_cfg: dict | None = None, rng_seed: int = 0):
    """Per-clip correctness + plan for one (policy, budget, scheme) on one codec."""
    budget = codec.budget_bits(bpe)
    reps = cfg["eval"]["random_draws"] if policy == "random" else 1
    outs, plan = [], None
    for r in range(reps):
        plan = codec.plan(scheme, policy, budget, task, seed=r)
        outs.append(codec.correct(task, plan, loss_cfg, rng_seed))
    return _avg(outs), plan


def _plan_record(codec, plan):
    return {"bits": plan.bits.tolist(), "copies": plan.copies.tolist(),
            "total_bits": plan.total(codec.group_sizes)}


# ------------------------------------------------------------------ E1
def run_e1(codecs, cfg):
    rows, perclip = [], {}
    for task in TASKS:
        sp = getattr(codecs[0].b, f"{task}_test")
        ref = accuracy_per_clip(predict(codecs[0].models[task], sp.x), sp.y)
        rows.append({"task": task, "policy": "original_features", **_ci(cfg, ref)})
        fl = _avg([c.correct(task, None, float_latent=True) for c in codecs])
        rows.append({"task": task, "policy": "autoencoder_unquantized", **_ci(cfg, fl)})
        for policy in cfg["policies"]:
            for bpe in cfg["budgets_bpe"]:
                res = [eval_policy(c, cfg, task, policy, bpe) for c in codecs]
                clip = _avg([r[0] for r in res])
                perclip[(task, policy, bpe)] = clip
                tb = res[0][1].total(codecs[0].group_sizes)
                rows.append({"task": task, "policy": policy, "bpe": bpe,
                             "kbps": kbps(tb, 1.0), "total_bits": tb, **_ci(cfg, clip),
                             "seed_acc": [float(r[0].mean()) for r in res],
                             "plans": [_plan_record(c, r[1]) for c, r in zip(codecs, res)]})
    paired = []
    for task in TASKS:
        for bpe in cfg["budgets_bpe"]:
            a = perclip[(task, f"task_{task}", bpe)]
            for other in ("variance", "uniform", "random", f"task_{'sid' if task == 'kws' else 'kws'}"):
                paired.append({"task": task, "bpe": bpe, "a": f"task_{task}", "b": other,
                               **_pd(cfg, a, perclip[(task, other, bpe)])})
    sens = [{k: v.tolist() for k, v in c.sens().items()} for c in codecs]
    return {"rows": rows, "paired": paired, "sensitivities": sens}, perclip


# ------------------------------------------------------------------ E2
def _combos(task):
    return [(f"task_{task}", "none"), (f"task_{task}", "eep"), (f"task_{task}", "uep_task"),
            (f"task_{task}", "uep_random"), ("variance", "none"), ("variance", "eep"),
            ("uniform", "none")]


def run_e2(codecs, cfg):
    rows, perclip = [], {}
    bpe = cfg["budget_bpe"]
    for task in TASKS:
        for policy, scheme in _combos(task):
            if scheme not in cfg["schemes"]:
                continue
            for model in cfg["loss_models"]:
                for p in cfg["loss_rates"]:
                    if p == 0.0 and model != cfg["loss_models"][0]:
                        perclip[(task, policy, scheme, model, p)] = perclip[(task, policy, scheme, cfg["loss_models"][0], p)]
                        continue
                    loss = {"model": model, "p": p, "burst_len": cfg["channel"]["burst_len"]}
                    res = [eval_policy(c, cfg, task, policy, bpe, scheme, loss,
                                       rng_seed=int(p * 1000) + (7 if model == "ge" else 0)) for c in codecs]
                    perclip[(task, policy, scheme, model, p)] = _avg([r[0] for r in res])
    for (task, policy, scheme, model, p), clip in perclip.items():
        base = perclip[(task, policy, scheme, cfg["loss_models"][0], 0.0)].mean()
        plan = codecs[0].plan(scheme, policy, codecs[0].budget_bits(bpe), task)
        rows.append({"task": task, "policy": policy, "scheme": scheme, "loss_model": model, "p": p,
                     "total_bits": plan.total(codecs[0].group_sizes), "source_bits": plan.source_budget,
                     "budget_bits": codecs[0].budget_bits(bpe), **_ci(cfg, clip),
                     "retained": float(clip.mean() / base) if base > 0 else None})
    paired = []
    for task in TASKS:
        a_key = (f"task_{task}", "uep_task")
        for model in cfg["loss_models"]:
            for p in cfg["loss_rates"]:
                if p == 0.0:
                    continue
                a = perclip.get((task, *a_key, model, p))
                if a is None:
                    continue
                for (policy, scheme) in [(f"task_{task}", "none"), (f"task_{task}", "eep"),
                                         (f"task_{task}", "uep_random"), ("variance", "eep"), ("uniform", "none")]:
                    b = perclip.get((task, policy, scheme, model, p))
                    if b is not None:
                        paired.append({"task": task, "loss_model": model, "p": p,
                                       "a": "task_aware+uep_task", "b": f"{policy}+{scheme}", **_pd(cfg, a, b)})
    # equal-bitrate check across schemes
    checks = []
    for task in TASKS:
        tb = {f"{pol}+{sch}": next(r["total_bits"] for r in rows if r["task"] == task and r["policy"] == pol
                                   and r["scheme"] == sch) for pol, sch in _combos(task) if sch in cfg["schemes"]}
        budget = codecs[0].budget_bits(bpe)
        checks.append({"task": task, "budget_bits": budget, "total_bits": tb,
                       "max_slack_bits": int(max(budget - v for v in tb.values())),
                       "group_quantum_bits": int(codecs[0].group_sizes.max()),
                       "pass": bool(all(0 <= budget - v < 2 * codecs[0].group_sizes.max() for v in tb.values()))})
    return {"rows": rows, "paired": paired, "equal_bitrate_check": checks}


# ------------------------------------------------------------------ E3
def run_e3(codecs, cfg):
    rows, paired = [], []
    for bpe in cfg["budgets_bpe"]:
        clips = {}
        for tuned in TASKS:
            for ev in TASKS:
                res = [eval_policy(c, cfg, ev, f"task_{tuned}", bpe) for c in codecs]
                clips[(tuned, ev)] = _avg([r[0] for r in res])
                rows.append({"bpe": bpe, "tuned_for": tuned, "evaluated_on": ev, **_ci(cfg, clips[(tuned, ev)])})
        for ev in TASKS:
            for ref in ("variance", "uniform"):
                res = [eval_policy(c, cfg, ev, ref, bpe) for c in codecs]
                clips[(ref, ev)] = _avg([r[0] for r in res])
                rows.append({"bpe": bpe, "tuned_for": ref, "evaluated_on": ev, **_ci(cfg, clips[(ref, ev)])})
        for ev in TASKS:
            other = "sid" if ev == "kws" else "kws"
            paired.append({"bpe": bpe, "evaluated_on": ev, "a": f"tuned_for_{ev}", "b": f"tuned_for_{other}",
                           **_pd(cfg, clips[(ev, ev)], clips[(other, ev)])})
    return {"rows": rows, "paired": paired}


# ------------------------------------------------------------------ E4
def run_e4(codecs, cfg):
    b = codecs[0].b
    a = cfg["audio"]
    rows = []
    ref_lm = b.norm.invert(b.kws_test.x)
    stoi_ok, stoi_note = True, None
    try:
        import pystoi  # noqa: F401
    except ImportError:
        stoi_ok, stoi_note = False, "pystoi not importable in this environment; STOI not computed"
    fb = mel_filterbank(a).to(b.device)
    pos = b.stoi_pos[: cfg["stoi_subset"]]
    wavs = b.stoi_wavs[: len(pos)]
    ceiling = None
    if stoi_ok:
        gl = logmel_to_wav(ref_lm[pos], a, fb=fb).cpu().numpy()
        ceiling = stoi_per_clip(wavs, gl)
    for policy in cfg["policies"]:
        for bpe in cfg["budgets_bpe"]:
            l1s, plans = [], []
            for c in codecs:
                plan = c.plan("none", policy, c.budget_bits(bpe), "kws")
                l1s.append(mel_l1_per_clip(ref_lm, c.recon_logmel("kws", plan)))
                plans.append(plan)
            row = {"policy": policy, "bpe": bpe, "kbps": kbps(plans[0].total(codecs[0].group_sizes), 1.0),
                   "mel_l1": _ci(cfg, _avg(l1s))}
            if stoi_ok:
                lm = codecs[0].recon_logmel("kws", plans[0])[pos]
                st = stoi_per_clip(wavs, logmel_to_wav(lm, a, fb=fb).cpu().numpy())
                row["stoi"] = _ci(cfg, st)
                row["stoi_n"] = int(len(pos))
            rows.append(row)
    c0 = codecs[0]
    x = b.kws_test.x[:1]
    cx = {"autoencoder_params": count_params(c0.ae), "encoder_params": count_params(c0.ae.encoder),
          "decoder_params": count_params(c0.ae.decoder),
          "encoder_macs": count_macs(c0.ae.encoder, x),
          "decoder_macs": count_macs(c0.ae.decoder, c0.z[("kws", "test")][:1]),
          "kws_params": count_params(b.kws), "kws_macs": count_macs(b.kws, x),
          "sid_params": count_params(b.sid), "sid_macs": count_macs(b.sid, x)}
    return {"rows": rows, "complexity": cx, "stoi_ceiling_griffin_lim": _ci(cfg, ceiling) if ceiling is not None else None,
            "stoi_note": stoi_note or "STOI is computed on Griffin-Lim vocoded audio, so absolute values are pessimistic"}


# ------------------------------------------------------------------ ablation
def run_ablation(codecs_by_g, cfg):
    rows, paired = [], []
    for G, codecs in codecs_by_g.items():
        for task in TASKS:
            for bpe in cfg["budgets_bpe"]:
                clips = {}
                for policy in cfg["policies"]:
                    res = [eval_policy(c, cfg, task, policy, bpe) for c in codecs]
                    clips[policy] = _avg([r[0] for r in res])
                    rows.append({"G": G, "task": task, "policy": policy, "bpe": bpe, **_ci(cfg, clips[policy])})
                for other in ("variance", "random", "uniform"):
                    if other in clips and f"task_{task}" in clips:
                        paired.append({"G": G, "task": task, "bpe": bpe, "a": f"task_{task}", "b": other,
                                       **_pd(cfg, clips[f"task_{task}"], clips[other])})
    return {"rows": rows, "paired": paired}


# ------------------------------------------------------------------ runner
def git_commit() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=Path(__file__).parent,
                                       stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:  # noqa: BLE001
        return None


def write_result(out_dir: Path, name: str, cfg: dict, body: dict, seeds, t0: float, counts: dict):
    out_dir.mkdir(parents=True, exist_ok=True)
    doc = {"experiment": name, "config_hash": config_hash(cfg), "config": cfg, "seeds": list(seeds),
           "git_commit": git_commit(), "wall_s": round(time.time() - t0, 1),
           "gpu_s": round(time.time() - t0, 1) if torch.cuda.is_available() else 0.0,
           "dataset_counts": counts, **body}
    with open(out_dir / f"{name}.json", "w", encoding="utf-8") as f:
        json.dump(doc, f, indent=1)


def run_experiment(name: str, fn, codecs, cfg: dict, out_dir: Path, seeds, counts: dict, log=print):
    t0 = time.time()
    try:
        res = fn(codecs, cfg)
        body = res[0] if isinstance(res, tuple) else res
        write_result(out_dir, name, cfg, body, seeds, t0, counts)
        log(f"{name} done in {time.time() - t0:.0f}s")
    except Exception as e:  # noqa: BLE001 - recorded, then re-raised
        out_dir.mkdir(parents=True, exist_ok=True)
        with open(out_dir / "FAILED.json", "a", encoding="utf-8") as f:
            json.dump({"experiment": name, "error": repr(e), "trace": traceback.format_exc()[-2000:]}, f)
            f.write("\n")
        raise
