"""Field visualization data adapters."""

from __future__ import annotations

import numpy as np


def central_slice(field: np.ndarray, axis: int = 2) -> np.ndarray:
    arr = np.asarray(field)
    idx = arr.shape[axis] // 2
    return np.take(arr, idx, axis=axis)
