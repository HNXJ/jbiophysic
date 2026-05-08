"""Fano-factor helpers."""

from __future__ import annotations

import numpy as np


def fano_factor(counts: np.ndarray, ddof: int = 1, eps: float = 1e-12) -> float:
    """Return variance-to-mean ratio for spike counts."""
    counts = np.asarray(counts, dtype=float)
    if counts.size == 0:
        raise ValueError("counts must not be empty")
    mean = float(np.mean(counts))
    if abs(mean) < eps:
        return 0.0
    return float(np.var(counts, ddof=ddof) / mean)
