"""Parameter sweep helpers."""

from __future__ import annotations

from collections.abc import Callable, Iterable


def run_scalar_sweep(values: Iterable[float], fn: Callable[[float], float]) -> list[tuple[float, float]]:
    """Evaluate `fn` for each scalar value."""
    return [(float(v), float(fn(float(v)))) for v in values]
