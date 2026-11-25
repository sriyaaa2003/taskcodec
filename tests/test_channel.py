import numpy as np
import torch

from taskcodec.channel import Plan, build_plan, received_mask, transmit
from taskcodec.quant import total_bits

SIZES = np.array([104] * 8)
SENS = np.array([8, 7, 6, 5, 4, 3, 2, 1.0])


def test_iid_rate():
    ok = received_mask(4000, 13, np.ones(8, dtype=int), {"model": "iid", "p": 0.2}, 1,
                       np.random.default_rng(0))
    assert abs((1 - ok.mean()) - 0.2) < 0.01


def test_ge_rate_and_burstiness():
    cfg = {"model": "ge", "p": 0.2, "burst_len": 4.0}
    ok = received_mask(6000, 13, np.ones(8, dtype=int), cfg, 1, np.random.default_rng(1))
    assert abs((1 - ok.mean()) - 0.2) < 0.02
    lost = ~ok[:, :, 0]
    cond = (lost[:, 1:] & lost[:, :-1]).sum() / lost[:, :-1].sum()
    assert cond > 0.5            # a loss is followed by another loss far more often than p


def test_repeated_class_loss_is_p_to_copies():
    p = 0.3
    copies = np.array([2, 2, 1, 1, 1, 1, 1, 1])
    ok = received_mask(8000, 13, copies, {"model": "iid", "p": p}, 1, np.random.default_rng(2))
    assert abs((1 - ok[:, :, 0].mean()) - p ** 2) < 0.01
    assert abs((1 - ok[:, :, 7].mean()) - p) < 0.01


def test_seed_reproducible():
    c = np.array([1, 2, 1, 1])
    a = received_mask(50, 13, c, {"model": "ge", "p": 0.3}, 1, np.random.default_rng(5))
    b = received_mask(50, 13, c, {"model": "ge", "p": 0.3}, 1, np.random.default_rng(5))
    assert (a == b).all()


def test_full_loss_and_no_loss():
    c = np.ones(4, dtype=int)
    assert received_mask(5, 13, c, {"p": 1.0}, 1, np.random.default_rng(0)).sum() == 0
    assert received_mask(5, 13, c, {"p": 0.0}, 1, np.random.default_rng(0)).all()


def test_concealment_is_stored_mean():
    z = torch.randn(30, 13, 4, 6) + 5.0
    mu = torch.zeros(4, 6)
    plan = Plan(np.array([2] * 4), np.ones(4, dtype=np.int64), "none", 0)
    rx = transmit(z, plan, {"model": "iid", "p": 0.5}, np.random.default_rng(0), mu, 1)
    same = torch.isclose(rx, z).all(dim=-1)
    lost = ~same
    assert lost.any() and same.any()
    assert (rx[lost] == 0).all()


def test_chunk_len_groups_steps():
    ok = received_mask(200, 13, np.ones(2, dtype=int), {"model": "iid", "p": 0.5}, 4,
                       np.random.default_rng(0))
    assert (ok[:, 0, 0] == ok[:, 3, 0]).all() and (ok[:, 4, 0] == ok[:, 7, 0]).all()


def test_equal_total_bits_across_schemes():
    budget = 1664
    plans = {s: build_plan(s, "task_kws", SENS, SENS, budget, SIZES)
             for s in ["none", "eep", "uep_task", "uep_random"]}
    for s, p in plans.items():
        tot = total_bits(p.bits, SIZES, p.copies)
        assert tot <= budget and budget - tot < 2 * SIZES.max(), (s, tot)
    assert (plans["eep"].copies == 2).all()
    assert plans["uep_task"].copies.sum() == plans["uep_random"].copies.sum() == 8 + 2
    assert set(np.flatnonzero(plans["uep_task"].copies == 2)) == {0, 1}
    assert plans["eep"].source_budget < plans["uep_task"].source_budget <= plans["none"].source_budget


def test_repetitions_are_counted_in_rate():
    p = build_plan("uep_task", "variance", SENS, SENS, 1664, SIZES)
    assert total_bits(p.bits, SIZES, p.copies) > total_bits(p.bits, SIZES)
