"""Stage drivers: prep (features + task models), AE training, bundle loading.

Kaggle notebooks are thin wrappers around these functions.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

from .data import (build_sid_subset, ensure_dataset, index_clips, load_wavs, official_split,
                   read_list)
from .features import Normalizer, mel_filterbank, wav_to_logmel
from .models import Autoencoder
from .tasks import TaskCNN, predict, train_classifier
from .train import train_autoencoder


def work(cfg) -> Path:
    p = Path(cfg["paths"]["work_dir"])
    p.mkdir(parents=True, exist_ok=True)
    return p


def run_prep(cfg: dict, device: str, log=print) -> dict:
    """Download data, cache log-mel features, train and freeze the KWS and SID models."""
    t0 = time.time()
    W = work(cfg)
    d, a = cfg["data"], cfg["audio"]
    root = ensure_dataset(cfg["paths"]["data_root"], d["urls"], d["expected_min_clips"]
                          if not d.get("max_clips") else 0)
    clips = index_clips(root)
    if d.get("max_clips"):
        rng = np.random.default_rng(cfg["seed"])
        keep = np.sort(rng.choice(len(clips), min(d["max_clips"], len(clips)), replace=False))
        clips = [clips[i] for i in keep]
    rels = [c["rel"] for c in clips]
    words = sorted({c["word"] for c in clips})
    word_idx = np.array([words.index(c["word"]) for c in clips], dtype=np.int64)
    speakers = [c["speaker"] for c in clips]
    split_off = official_split(rels, read_list(root / "validation_list.txt"),
                               read_list(root / "testing_list.txt"))
    sid_label, sid_split = build_sid_subset(speakers, d["n_speakers"], d["sid_split"], cfg["seed"])
    log(f"clips={len(clips)} words={len(words)} official split counts="
        f"{np.bincount(split_off, minlength=3).tolist()}")

    fb = mel_filterbank(a).to(device)
    feats = np.empty((len(clips), a["n_mels"], a["frames"]), dtype=np.float16)
    bs = 2048
    for i in range(0, len(clips), bs):
        wav = load_wavs(root, rels[i:i + bs], a["clip_samples"])
        feats[i:i + bs] = wav_to_logmel(torch.from_numpy(wav).to(device), a, fb).cpu().numpy()
        if (i // bs) % 10 == 0:
            log(f"  features {i}/{len(clips)}")
    np.save(W / "logmel.npy", feats)

    tr = np.flatnonzero(split_off == 0)
    nz = Normalizer().fit(torch.from_numpy(feats[tr[:40000]].astype(np.float32)))
    torch.save(nz.state(), W / "norm.pt")

    # STOI subset: waveforms of seeded KWS test clips
    te = np.flatnonzero(split_off == 2)
    rng = np.random.default_rng(cfg["seed"] + 1)
    pos = np.sort(rng.choice(len(te), min(d["stoi_subset"], len(te)), replace=False))
    stoi_idx = te[pos]
    np.save(W / "stoi_wavs.npy", load_wavs(root, [rels[i] for i in stoi_idx], a["clip_samples"]))
    np.save(W / "stoi_idx.npy", stoi_idx)

    np.savez(W / "meta.npz", word_idx=word_idx, split_off=split_off, sid_label=sid_label,
             sid_split=sid_split, words=np.array(words), rels=np.array(rels))

    # task models
    xn = lambda idx: nz.apply(torch.from_numpy(feats[idx].astype(np.float32)).to(device))  # noqa: E731
    report = {"n_clips": len(clips), "n_words": len(words),
              "kws_counts": {k: int((split_off == i).sum()) for i, k in enumerate(["train", "val", "test"])},
              "n_speakers": int(sid_label.max() + 1),
              "sid_counts": {k: int((sid_split == i).sum()) for i, k in enumerate(["train", "val", "test"])}}
    ytr = torch.from_numpy(word_idx).to(device)
    kws = TaskCNN(len(words), cfg["tasks"]["channels"], cfg["tasks"]["dropout"]).to(device)
    tr_i, va_i, te_i = (np.flatnonzero(split_off == s) for s in (0, 1, 2))
    log("training KWS")
    report["kws_train"] = train_classifier(kws, xn(tr_i), ytr[tr_i], xn(va_i), ytr[va_i], cfg, cfg["seed"], log)
    report["kws_val_acc"] = float((predict(kws, xn(va_i)).argmax(1) == ytr[va_i]).float().mean())
    report["kws_test_acc"] = float((predict(kws, xn(te_i)).argmax(1) == ytr[te_i]).float().mean())
    torch.save(kws.state_dict(), W / "kws.pt")

    ys = torch.from_numpy(sid_label).to(device)
    s_tr, s_va, s_te = (np.flatnonzero(sid_split == s) for s in (0, 1, 2))
    sid = TaskCNN(report["n_speakers"], cfg["tasks"]["channels"], cfg["tasks"]["dropout"]).to(device)
    log("training SID")
    report["sid_train"] = train_classifier(sid, xn(s_tr), ys[s_tr], xn(s_va), ys[s_va], cfg, cfg["seed"], log)
    report["sid_val_acc"] = float((predict(sid, xn(s_va)).argmax(1) == ys[s_va]).float().mean())
    report["sid_test_acc"] = float((predict(sid, xn(s_te)).argmax(1) == ys[s_te]).float().mean())
    torch.save(sid.state_dict(), W / "sid.pt")
    report["wall_s"] = time.time() - t0
    with open(W / "task_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    log(json.dumps(report, indent=2))
    return report


@dataclass
class Split:
    x: torch.Tensor          # normalised log-mel
    y: torch.Tensor
    idx: np.ndarray          # row positions in the global arrays


@dataclass
class Bundle:
    cfg: dict
    norm: Normalizer
    kws: TaskCNN
    sid: TaskCNN
    ae_train: torch.Tensor
    kws_val: Split
    kws_test: Split
    sid_val: Split
    sid_test: Split
    stoi_wavs: np.ndarray
    stoi_pos: np.ndarray     # positions of the STOI clips inside kws_test
    counts: dict = field(default_factory=dict)
    device: str = "cpu"


def load_bundle(cfg: dict, device: str, work_dir: Path | None = None) -> Bundle:
    W = Path(work_dir) if work_dir else work(cfg)
    feats = np.load(W / "logmel.npy")
    meta = np.load(W / "meta.npz")
    nz = Normalizer.from_state(torch.load(W / "norm.pt"))
    a = cfg["audio"]
    split_off, word_idx = meta["split_off"], meta["word_idx"]
    sid_label, sid_split = meta["sid_label"], meta["sid_split"]
    n_words, n_spk = len(meta["words"]), int(sid_label.max() + 1)

    def mk(idx, labels):
        x = nz.apply(torch.from_numpy(feats[idx].astype(np.float32)).to(device))
        return Split(x, torch.from_numpy(labels[idx]).to(device), idx)

    kws = TaskCNN(n_words, cfg["tasks"]["channels"], cfg["tasks"]["dropout"]).to(device)
    kws.load_state_dict(torch.load(W / "kws.pt", map_location=device))
    sid = TaskCNN(n_spk, cfg["tasks"]["channels"], cfg["tasks"]["dropout"]).to(device)
    sid.load_state_dict(torch.load(W / "sid.pt", map_location=device))
    for m in (kws, sid):
        m.eval()
        for p in m.parameters():
            p.requires_grad_(False)

    # AE trains on official-train clips that are not SID val/test clips
    ae_idx = np.flatnonzero((split_off == 0) & ~np.isin(sid_split, [1, 2]))
    kws_test = mk(np.flatnonzero(split_off == 2), word_idx)
    stoi_idx = np.load(W / "stoi_idx.npy")
    pos = np.searchsorted(kws_test.idx, stoi_idx)
    b = Bundle(cfg, nz, kws, sid,
               nz.apply(torch.from_numpy(feats[ae_idx].astype(np.float32)).to(device)),
               mk(np.flatnonzero(split_off == 1), word_idx), kws_test,
               mk(np.flatnonzero(sid_split == 1), sid_label), mk(np.flatnonzero(sid_split == 2), sid_label),
               np.load(W / "stoi_wavs.npy"), pos, device=device)
    b.counts = {"ae_train": int(len(ae_idx)), "kws_val": len(b.kws_val.idx), "kws_test": len(kws_test.idx),
                "sid_val": len(b.sid_val.idx), "sid_test": len(b.sid_test.idx), "n_words": n_words,
                "n_speakers": n_spk, "stoi_clips": len(pos)}
    return b


def ae_path(cfg, groups: int, seed: int) -> Path:
    return work(cfg) / f"ae_G{groups}_s{seed}.pt"


def run_train_ae(cfg: dict, bundle: Bundle, groups_list, seeds_for, log=print) -> dict:
    """Train one AE per (G, seed) and save it. seeds_for(G) -> list of seeds."""
    out = {}
    for G in groups_list:
        for s in seeds_for(G):
            p = ae_path(cfg, G, s)
            if p.exists():
                log(f"skip existing {p.name}")
                continue
            ae, info = train_autoencoder(bundle.ae_train, bundle.norm, cfg, G, s, log,
                                     task_models=[bundle.kws, bundle.sid])
            torch.save(ae.state_dict(), p)
            out[f"G{G}_s{s}"] = {"final_loss": info["loss_history"][-1], "train_s": info["train_seconds"]}
    return out


def load_ae(cfg: dict, groups: int, seed: int, device: str) -> Autoencoder:
    a, lat = cfg["audio"], cfg["latent"]
    ae = Autoencoder(a["n_mels"], a["frames"], groups, lat["total_dim"], cfg["ae"]["channels"]).to(device)
    ae.load_state_dict(torch.load(ae_path(cfg, groups, seed), map_location=device))
    ae.eval()
    for p in ae.parameters():
        p.requires_grad_(False)
    return ae
