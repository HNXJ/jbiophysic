"""Network builder scaffolds."""

from __future__ import annotations

from dataclasses import dataclass

from .populations import PopulationSpec


@dataclass(frozen=True)
class NetworkSpec:
    populations: tuple[PopulationSpec, ...]

    @property
    def n_neurons(self) -> int:
        return sum(pop.count for pop in self.populations)


def make_ei_network(n_e: int = 40, n_i: int = 10) -> NetworkSpec:
    """Create a minimal E/I population specification."""
    return NetworkSpec(
        populations=(
            PopulationSpec("E", n_e, "excitatory"),
            PopulationSpec("I", n_i, "inhibitory"),
        )
    )
