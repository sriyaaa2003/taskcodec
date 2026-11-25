"""Per-clip metrics. All take tensors/arrays and return numbers or per-clip arrays."""
from __future__ import annotations

import numpy as np
import torch


def accuracy_per_clip(logits: torch.Tensor, y: torch.Tensor) -> np.ndarray:
    return (logits.argmax(dim=1) == y).float().cpu().numpy()


def mel_l1_per_clip(logmel_ref: torch.Tensor, logmel_hat: torch.Tensor) -> np.ndarray:
    """Mean absolute error in natural-log mel units (un-normalised features)."""
    return (logmel_ref - logmel_hat).abs().mean(dim=(1, 2)).cpu().numpy()


def stoi_per_clip(ref_wavs: np.ndarray, deg_wavs: np.ndarray, sr: int = 16000) -> np.ndarray:
    """STOI per clip (pystoi). Raises ImportError if pystoi is missing."""
    from pystoi import stoi
    return np.array([stoi(r, d, sr, extended=False) for r, d in zip(ref_wavs, deg_wavs)])
