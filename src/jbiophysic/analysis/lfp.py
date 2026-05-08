"""LFP/CSD summaries."""

from __future__ import annotations

import numpy as np


def csd_layer_summary(csd: np.ndarray, axis: int = 2) -> np.ndarray:
    """Average CSD over all axes except the requested laminar/depth axis."""
    arr = np.asarray(csd, dtype=float)
    axes = tuple(i for i in range(arr.ndim) if i != axis)
    return np.mean(arr, axis=axes)


def lfp_rms(lfp: np.ndarray, axis: int | None = None) -> np.ndarray:
    """Root-mean-square LFP amplitude."""
    arr = np.asarray(lfp, dtype=float)
    return np.sqrt(np.mean(arr**2, axis=axis))
