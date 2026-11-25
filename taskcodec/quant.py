"""Per-group uniform scalar quantizer with straight-through gradient, and bit accounting."""
from __future__ import annotations

import numpy as np
import torch


def _expand_bits(bits, z: torch.Tensor) -> torch.Tensor:
    """bits: (G,) shared across clips, or (B, G) per clip -> broadcastable to z (B, T, G, d)."""
    b = torch.as_tensor(bits, dtype=z.dtype, device=z.device)
    if b.ndim == 0:
        return b
    if b.ndim == 1:
        return b[:, None]
    if b.ndim == 2:
        return b[:, None, :, None]
    raise ValueError(f"bits must have 0, 1 or 2 dims, got {b.ndim}")


def quantize(z: torch.Tensor, bits, mu: torch.Tensor, sigma: torch.Tensor,
             clip_k: float = 3.0, ste: bool = True) -> torch.Tensor:
    """Midrise uniform quantizer on [mu - k*sigma, mu + k*sigma] with 2**bits levels per group.

    0 bits -> one level at the centre, i.e. the stored mean `mu`.
    Does not decide the bits; the caller (alloc.py) does.
    """
    b = _expand_bits(bits, z)
    levels = torch.exp2(b)
    half = clip_k * sigma
    u = ((z - mu) / half).clamp(-1.0, 1.0)
    idx = torch.floor((u + 1.0) * 0.5 * levels)
    idx = torch.minimum(idx, levels - 1.0)
    q = ((idx + 0.5) / levels * 2.0 - 1.0) * half + mu
    if ste:
        return z + (q - z).detach()
    return q


def total_bits(bits, group_sizes, copies=None) -> int:
    """bits = sum_g n_g * b_g * copies_g (this is the number reported as the rate)."""
    b = np.asarray(bits, dtype=np.int64)
    n = np.asarray(group_sizes, dtype=np.int64)
    c = np.ones_like(b) if copies is None else np.asarray(copies, dtype=np.int64)
    return int((n * b * c).sum())


def kbps(bits: int, duration_s: float) -> float:
    return bits / duration_s / 1000.0
