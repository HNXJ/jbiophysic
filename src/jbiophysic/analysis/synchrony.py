"""Simple synchrony summaries."""

from __future__ import annotations

import numpy as np


def mean_pairwise_correlation(signals: np.ndarray) -> float:
    """Mean upper-triangle correlation for `[time, units]` signals."""
    x = np.asarray(signals, dtype=float)
    if x.ndim != 2 or x.shape[1] < 2:
        raise ValueError("signals must be [time, units] with at least two units")
    corr = np.corrcoef(x, rowvar=False)
    iu = np.triu_indices(corr.shape[0], k=1)
    return float(np.nanmean(corr[iu]))
