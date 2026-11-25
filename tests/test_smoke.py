"""End-to-end smoke test on tiny synthetic data (CPU). Checks plumbing, not accuracy."""
import numpy as np
import torch

from taskcodec import experiments as ex
from taskcodec.config import load_config
from taskcodec.features import Normalizer
from taskcodec.pipeline import Bundle, Split
from taskcodec.tasks import TaskCNN, train_classifier
from taskcodec.train import train_autoencoder


def _cfg():
    cfg = load_config("e1")
    cfg["tasks"].update(epochs=2, batch_size=32, channels=[8, 8, 8, 8])
    cfg["ae"].update(epochs=1, batch_size=32, channels=[8, 8, 8], stats_clips=64)
    cfg["eval"].update(bootstrap=100, random_draws=2, batch_size=64)
    cfg["channel"]["draws"] = 2
    cfg["sensitivity"]["max_clips"] = 64
    return cfg


def _bundle(cfg):
    g = torch.Generator().manual_seed(0)
    mk = lambda n, k: Split(torch.randn(n, 40, 104, generator=g), torch.randint(0, k, (n,), generator=g), np.arange(n))  # noqa: E731
    kws_tr, sid_tr = mk(96, 5), mk(96, 4)
    nz = Normalizer().fit(kws_tr.x)
    kws = TaskCNN(5, cfg["tasks"]["channels"]).eval()
    sid = TaskCNN(4, cfg["tasks"]["channels"]).eval()
    train_classifier(kws, kws_tr.x, kws_tr.y, kws_tr.x, kws_tr.y, cfg, 0, log=lambda *_: None)
    train_classifier(sid, sid_tr.x, sid_tr.y, sid_tr.x, sid_tr.y, cfg, 0, log=lambda *_: None)
    return Bundle(cfg, nz, kws, sid, kws_tr.x, mk(64, 5), mk(80, 5), mk(48, 4), mk(48, 4),
                  np.random.randn(6, 16000).astype(np.float32) * 0.1, np.arange(6))


def _codecs(cfg, bundle, groups, seeds):
    out = []
    for s in seeds:
        ae, _ = train_autoencoder(bundle.ae_train, bundle.norm, cfg, groups, s, log=lambda *_: None, task_models=[bundle.kws, bundle.sid])
        out.append(ex.Codec(ae, bundle, cfg, groups, s))
    return out


def test_pipeline_end_to_end(tmp_path):
    cfg = _cfg()
    b = _bundle(cfg)
    codecs = _codecs(cfg, b, 8, [0])
    c = codecs[0]
    s = c.sens()
    assert set(s) == {"variance", "task_kws", "task_sid"} and all(np.isfinite(v).all() and len(v) == 8 for v in s.values())

    e1 = dict(cfg, budgets_bpe=[1.0, 2.0], policies=["uniform", "variance", "task_kws", "task_sid", "random"])
    r1, _ = ex.run_e1(codecs, e1)
    assert {r["policy"] for r in r1["rows"]} >= {"uniform", "random", "original_features"}
    assert all(0.0 <= r["mean"] <= 1.0 and r["lo"] <= r["hi"] for r in r1["rows"])
    assert len(r1["paired"]) == 2 * 2 * 4

    e2 = dict(cfg, budget_bpe=2.0, loss_rates=[0.0, 0.3], loss_models=["iid", "ge"],
              schemes=["none", "eep", "uep_task", "uep_random"])
    r2 = ex.run_e2(codecs, e2)
    assert all(ck["pass"] for ck in r2["equal_bitrate_check"])
    assert any(p["b"].endswith("eep") for p in r2["paired"])

    r3 = ex.run_e3(codecs, dict(cfg, budgets_bpe=[1.0]))
    assert len([r for r in r3["rows"] if r["tuned_for"] in ("kws", "sid")]) == 4

    e4 = dict(cfg, budgets_bpe=[1.0], policies=["uniform", "task_kws"], stoi_subset=3)
    r4 = ex.run_e4(codecs, e4)
    assert r4["complexity"]["encoder_macs"] > 0 and r4["rows"][0]["mel_l1"]["mean"] > 0

    ab = ex.run_ablation({8: codecs}, dict(cfg, budgets_bpe=[1.0], policies=["uniform", "variance", "task_kws", "task_sid", "random"]))
    assert ab["paired"]

    ex.run_experiment("e1", lambda cs, cf: ex.run_e1(cs, cf), codecs, e1, tmp_path, [0], b.counts)
    assert (tmp_path / "e1.json").exists()
