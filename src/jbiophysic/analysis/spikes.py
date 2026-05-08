"""Spike-train statistics."""

from __future__ import annotations

import numpy as np


def threshold_crossings(voltage: np.ndarray, threshold: float = 0.0) -> np.ndarray:
    """Boolean spike vector from upward threshold crossings."""
    v = np.asarray(voltage)
    if v.ndim != 1:
        raise ValueError("voltage must be one-dimensional")
    return np.concatenate([[False], (v[1:] >= threshold) & (v[:-1] < threshold)])


def firing_rate_hz(spikes: np.ndarray, dt_s: float) -> float:
    """Return firing rate in Hz."""
    if dt_s <= 0:
        raise ValueError("dt_s must be positive")
    spikes = np.asarray(spikes, dtype=bool)
    duration_s = spikes.size * dt_s
    return float(spikes.sum() / duration_s) if duration_s > 0 else 0.0


def spike_counts(spike_matrix: np.ndarray, axis: int = -1) -> np.ndarray:
    """Count spikes over an axis."""
    return np.sum(np.asarray(spike_matrix, dtype=bool), axis=axis)
