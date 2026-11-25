"""Group sensitivity estimation and bit-allocation policies.

Policies: uniform, variance (reverse water-filling), task_<name> (sensitivity greedy), random.
All return integer bits per group with sum_g n_g * b_g <= budget.
"""
from __future__ import annotations

from typing import Callable

import numpy as np
import torch
import torch.nn.functional as F


class InfeasibleBudget(ValueError):
    pass


def allocate(policy: str, sens, budget: int, group_sizes, b_min: int = 0,
             b_max: int = 8, seed: int = 0) -> np.ndarray:
    """Marginal-return greedy: give one bit to the group with the largest s_g^2 * 2^(-2 b_g).

    `sens` is a per-group scale s_g (task sensitivity, or sigma for `variance`); it is ignored
    by `uniform` and `random`. Ties go to the lowest group index. Raises if the budget cannot
    pay for b_min bits everywhere.
    """
    sizes = np.asarray(group_sizes, dtype=np.int64)
    G = len(sizes)
    if budget < b_min * sizes.sum():
        raise InfeasibleBudget(f"budget {budget} < minimum {b_min * sizes.sum()}")
    bits = np.full(G, b_min, dtype=np.int64)
    remaining = int(budget - b_min * sizes.sum())

    if policy == "uniform":
        s = np.ones(G)
    elif policy == "random":
        s = None
    elif policy == "variance" or policy.startswith("task_"):
        if sens is None:
            raise ValueError(f"policy {policy} needs sensitivities")
        s = np.asarray(sens, dtype=np.float64)
        if s.shape != (G,):
            raise ValueError(f"sens shape {s.shape} != ({G},)")
    else:
        raise ValueError(f"unknown policy {policy}")

    rng = np.random.default_rng(seed) if policy == "random" else None
    while True:
        eligible = (bits < b_max) & (sizes <= remaining)
        if not eligible.any():
            break
        if rng is not None:
            g = int(rng.choice(np.flatnonzero(eligible)))
        else:
            prio = np.where(eligible, s ** 2 * np.exp2(-2.0 * bits), -np.inf)
            g = int(np.argmax(prio))
        bits[g] += 1
        remaining -= int(sizes[g])
    return bits


def variance_sensitivity(sigma: torch.Tensor) -> np.ndarray:
    """sigma: (G, d) per-element std -> RMS per group (reconstruction-optimal scale)."""
    return torch.sqrt((sigma ** 2).mean(dim=1)).cpu().numpy().astype(np.float64)


def group_sensitivity(decode_fn: Callable[[torch.Tensor], torch.Tensor],
                      task_fn: Callable[[torch.Tensor], torch.Tensor],
                      z_val: torch.Tensor, y_val: torch.Tensor, sigma: torch.Tensor,
                      max_clips: int = 4000, batch_size: int = 512) -> np.ndarray:
    """s_g = RMS(dL_task/dz_g) * sigma_g, estimated on VALIDATION latents only.

    The signature accepts only validation tensors on purpose (no test data can be passed by
    name). decode_fn and task_fn must be in eval mode.
    """
    n = min(len(z_val), max_clips)
    acc = None
    cnt = 0
    for i in range(0, n, batch_size):
        z = z_val[i:i + batch_size].detach().clone().requires_grad_(True)
        y = y_val[i:i + batch_size]
        loss = F.cross_entropy(task_fn(decode_fn(z)), y, reduction="sum")
        (g,) = torch.autograd.grad(loss, z)
        sq = (g ** 2).sum(dim=(0, 1, 3))            # (G,)
        acc = sq if acc is None else acc + sq
        cnt += g.shape[0] * g.shape[1] * g.shape[3]
    rms = torch.sqrt(acc / cnt)
    sig = torch.sqrt((sigma ** 2).mean(dim=1))
    return (rms * sig).detach().cpu().numpy().astype(np.float64)
