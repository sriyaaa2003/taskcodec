"""Packetisation, loss models, protection plans and concealment.

A packet carries one time chunk of one priority class (groups with the same number of copies).
A class with c copies is received if at least one of its c copies arrives; copies travel on
independent loss processes. Lost elements are replaced by the stored mean latent.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from .alloc import allocate
from .quant import total_bits


@dataclass(frozen=True)
class Plan:
    bits: np.ndarray        # (G,) bits per group
    copies: np.ndarray      # (G,) transmissions per group
    scheme: str
    source_budget: int      # budget given to the allocator

    def total(self, group_sizes) -> int:
        return total_bits(self.bits, group_sizes, self.copies)


def build_plan(scheme: str, policy: str, sens_alloc, sens_prot, budget: int, group_sizes,
               b_min: int = 0, b_max: int = 8, protect_frac: float = 0.25, seed: int = 0) -> Plan:
    """Allocate source bits so that the TOTAL channel bits (including repetitions) fit `budget`.

    none: one copy. eep: every group twice. uep_task: top-sensitivity groups (by sens_prot)
    twice. uep_random: random groups twice (same count). All are matched to the same total.
    """
    sizes = np.asarray(group_sizes, dtype=np.int64)
    G = len(sizes)
    copies = np.ones(G, dtype=np.int64)
    if scheme == "eep":
        copies[:] = 2
    elif scheme in ("uep_task", "uep_random"):
        m = max(1, int(round(protect_frac * G)))
        if scheme == "uep_task":
            if sens_prot is None:
                raise ValueError("uep_task needs sens_prot")
            order = np.argsort(-np.asarray(sens_prot), kind="stable")
            copies[order[:m]] = 2
        else:
            rng = np.random.default_rng(seed + 12345)
            copies[rng.choice(G, size=m, replace=False)] = 2
    elif scheme != "none":
        raise ValueError(f"unknown scheme {scheme}")

    src = int(budget)
    while True:
        bits = allocate(policy, sens_alloc, src, sizes, b_min, b_max, seed)
        tot = total_bits(bits, sizes, copies)
        if tot <= budget:
            return Plan(bits, copies, scheme, src)
        src -= max(1, -(-(tot - budget) // int(copies.max())))
        if src < b_min * sizes.sum():
            from .alloc import InfeasibleBudget
            raise InfeasibleBudget(f"{scheme}: cannot fit repetitions into budget {budget}")


def _loss_stream(n: int, length: int, loss_cfg: dict, rng: np.random.Generator) -> np.ndarray:
    """(n, length) boolean, True = packet lost. Models: iid, ge (Gilbert, good=0 loss/bad=1)."""
    p = float(loss_cfg.get("p", 0.0))
    if p <= 0.0:
        return np.zeros((n, length), dtype=bool)
    if p >= 1.0:
        return np.ones((n, length), dtype=bool)
    model = loss_cfg.get("model", "iid")
    if model == "iid":
        return rng.random((n, length)) < p
    if model == "ge":
        burst = float(loss_cfg.get("burst_len", 3.0))
        p_bg = 1.0 / burst                          # bad -> good
        p_gb = p / (1.0 - p) * p_bg                 # good -> bad, stationary P(bad) = p
        p_gb = min(p_gb, 1.0)
        state = rng.random(n) < p
        out = np.zeros((n, length), dtype=bool)
        for t in range(length):
            out[:, t] = state
            u = rng.random(n)
            state = np.where(state, u >= p_bg, u < p_gb)
        return out
    raise ValueError(f"unknown loss model {model}")


def received_mask(n_clips: int, n_steps: int, copies, loss_cfg: dict, chunk_len: int,
                  rng: np.random.Generator) -> np.ndarray:
    """(n_clips, n_steps, G) boolean, True = this group at this time step arrives."""
    copies = np.asarray(copies, dtype=np.int64)
    classes = np.unique(copies)                      # ascending copy counts
    class_of = np.searchsorted(classes, copies)      # (G,)
    n_chunks = -(-n_steps // chunk_len)
    streams = int(copies.max())
    seq_len = n_chunks * len(classes)
    lost = np.stack([_loss_stream(n_clips, seq_len, loss_cfg, rng) for _ in range(streams)])
    lost = lost.reshape(streams, n_clips, n_chunks, len(classes))
    cell_ok = np.zeros((n_clips, n_chunks, len(classes)), dtype=bool)
    for ci, c in enumerate(classes):
        ok = ~lost[:int(c), :, :, ci]                # (c, n, chunks)
        cell_ok[:, :, ci] = ok.any(axis=0)
    step_chunk = np.arange(n_steps) // chunk_len
    return cell_ok[:, step_chunk, :][:, :, class_of]


def transmit(latent_q: torch.Tensor, plan: Plan, loss_cfg: dict, rng: np.random.Generator,
             mean: torch.Tensor, chunk_len: int = 1) -> torch.Tensor:
    """latent_q: (N, T, G, d) quantized latent -> received latent (lost parts = `mean`).

    Never changes the bitrate: that is fixed by `plan`.
    """
    n, t, g, _ = latent_q.shape
    ok = received_mask(n, t, plan.copies, loss_cfg, chunk_len, rng)
    ok_t = torch.from_numpy(ok).to(latent_q.device)[..., None]
    return torch.where(ok_t, latent_q, mean.expand_as(latent_q))
