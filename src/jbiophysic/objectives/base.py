from dataclasses import dataclass
from typing import Any, Callable, Union

@dataclass(frozen=True)
class Objective:
    """An objective defines a metric and a target value for optimization."""
    metric_fn: Union[Callable, str]
    target: Any
    weight: float = 1.0
    name: str | None = None
