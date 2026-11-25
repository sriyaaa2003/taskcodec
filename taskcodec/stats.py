"""Bootstrap confidence intervals over test clips (never over batches)."""
from __future__ import annotations

import numpy as np


def _boot_means(x: np.ndarray, n_boot: int, seed: int, chunk: int = 500) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n = len(x)
    out = np.empty(n_boot)
    for i in range(0, n_boot, chunk):
        m = min(chunk, n_boot - i)
        idx = rng.integers(0, n, size=(m, n))
        out[i:i + m] = x[idx].mean(axis=1)
    return out


def bootstrap_ci(scores, n_boot: int = 10000, ci: float = 0.95, seed: int = 0):
    """Return (mean, lo, hi) of per-clip scores."""
    x = np.asarray(scores, dtype=np.float64)
    if x.ndim != 1 or len(x) == 0:
        raise ValueError("scores must be a non-empty 1-D array of per-clip values")
    b = _boot_means(x, n_boot, seed)
    a = (1.0 - ci) / 2.0
    return float(x.mean()), float(np.quantile(b, a)), float(np.quantile(b, 1.0 - a))


def paired_bootstrap(a, b, n_boot: int = 10000, ci: float = 0.95, seed: int = 0):
    """CI of mean(a - b) with the same clips resampled for both."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError("paired scores must have the same shape")
    return bootstrap_ci(a - b, n_boot, ci, seed)
