"""Run manifest objects for reproducible optimizer tests."""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class OptimizerManifest:
    optimizer: str
    root_seed: int
    batch_size: int
    max_steps: int
    alpha: float | None = None
    truth_mode: str = "truth_safe_unverified"

    def to_dict(self) -> dict[str, object]:
        return asdict(self)
