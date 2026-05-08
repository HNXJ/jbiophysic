"""Spectrum data adapters."""

from __future__ import annotations

import numpy as np


def spectrum_dict(freq_hz: np.ndarray, power: np.ndarray) -> dict[str, np.ndarray]:
    return {"frequency_hz": np.asarray(freq_hz), "power": np.asarray(power)}
