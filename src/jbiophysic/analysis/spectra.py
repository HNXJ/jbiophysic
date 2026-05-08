"""Spectral summaries for neural traces."""

from __future__ import annotations

import numpy as np
from scipy.signal import welch


def power_spectrum(x: np.ndarray, fs_hz: float, nperseg: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Return Welch frequency and power arrays."""
    if fs_hz <= 0:
        raise ValueError("fs_hz must be positive")
    x = np.asarray(x, dtype=float)
    if x.ndim != 1:
        raise ValueError("x must be one-dimensional")
    if nperseg is None:
        nperseg = min(256, x.size)
    return welch(x, fs=fs_hz, nperseg=nperseg)


def band_power(freq_hz: np.ndarray, power: np.ndarray, band: tuple[float, float]) -> float:
    """Integrate spectral power inside a frequency band."""
    lo, hi = band
    if lo < 0 or hi <= lo:
        raise ValueError("band must be nonnegative and increasing")
    mask = (freq_hz >= lo) & (freq_hz <= hi)
    if not np.any(mask):
        return 0.0
    return float(np.trapezoid(power[mask], freq_hz[mask]))


def beta_gamma_ratio(freq_hz: np.ndarray, power: np.ndarray, beta: tuple[float, float] = (13.0, 30.0), gamma: tuple[float, float] = (40.0, 100.0), eps: float = 1e-12) -> float:
    """Return beta/gamma power ratio."""
    return band_power(freq_hz, power, beta) / (band_power(freq_hz, power, gamma) + eps)
