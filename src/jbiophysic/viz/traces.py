"""Visualization data adapters; plotting backends stay optional."""

from __future__ import annotations

import numpy as np


def trace_dataframe_dict(time: np.ndarray, value: np.ndarray, name: str = "trace") -> dict[str, np.ndarray | str]:
    return {"time": np.asarray(time), "value": np.asarray(value), "name": name}
