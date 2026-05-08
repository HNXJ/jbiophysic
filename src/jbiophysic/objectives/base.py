from collections.abc import Callable
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class Objective:
    """An objective defines a metric and a target value for optimization."""
    metric_fn: Callable | str
    target: Any
    weight: float = 1.0
    name: str | None = None
