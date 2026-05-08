"""Raster data adapters."""

from __future__ import annotations

import numpy as np


def spike_indices(spike_matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return np.nonzero(np.asarray(spike_matrix, dtype=bool))
