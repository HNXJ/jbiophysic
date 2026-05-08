"""Population labels and counts."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PopulationSpec:
    name: str
    count: int
    cell_class: str

    def __post_init__(self) -> None:
        if self.count <= 0:
            raise ValueError("count must be positive")
