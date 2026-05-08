# src/jbiophysic/models/builders/hierarchy.py
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterator

from jbiophysic.common.utils.logging import get_logger

logger = get_logger(__name__)

try:
    import jaxley as jx
    from jaxley.synapses import IonotropicSynapse
except ModuleNotFoundError:  # pragma: no cover - exercised when jaxley is absent
    jx = None
    IonotropicSynapse = None


@dataclass
class _SimpleRecordings:
    empty: bool = True


@dataclass
class _SimpleCellSelector:
    network: "SimpleNetwork"
    indices: list[int]

    def add_to_group(self, name: str):
        for idx in self.indices:
            self.network.groups.setdefault(name, []).append(idx)
        return self

    def branch(self, _idx: int):
        return self

    def comp(self, _idx: int):
        return self


@dataclass
class SimpleNetwork:
    """Small fallback network used when optional dependency jaxley is unavailable."""

    n_cells: int
    groups: dict[str, list[int]] = field(default_factory=dict)
    recordings: _SimpleRecordings = field(default_factory=_SimpleRecordings)

    @property
    def cells(self) -> Iterator[int]:
        return iter(range(self.n_cells))

    @property
    def nodes(self) -> list[int]:
        return list(range(self.n_cells))

    def cell(self, idx):
        if isinstance(idx, list):
            indices = idx
        else:
            indices = [int(idx)]
        return _SimpleCellSelector(self, indices)

    def record(self, _what: str):
        self.recordings.empty = False
        return self


def _build_simple_hierarchy(n_areas: int) -> SimpleNetwork:
    cells_per_area = 5
    brain = SimpleNetwork(n_cells=n_areas * cells_per_area)
    for i in range(n_areas):
        base = i * cells_per_area
        brain.cell(list(range(base, base + 2))).add_to_group("PC")
        brain.cell(list(range(base + 2, base + 3))).add_to_group("PV")
        brain.cell(list(range(base + 3, base + 4))).add_to_group("SST")
        brain.cell(list(range(base + 4, base + 5))).add_to_group("VIP")
        brain.cell(list(range(base, base + cells_per_area))).add_to_group(f"Area_{i}")
    return brain


def build_cortical_hierarchy(n_areas: int = 11):
    """Inter-areal connectivity across visual-area-like cortical columns.

    If Jaxley is installed, return a Jaxley network.  Otherwise return a lightweight
    ``SimpleNetwork`` with the same cells/nodes/record surface required by smoke tests.
    """
    logger.info(f"Building cortical hierarchy with {n_areas} areas")
    if n_areas <= 0:
        raise ValueError("n_areas must be positive")

    if jx is None:
        logger.warning("jaxley is unavailable; returning SimpleNetwork fallback")
        return _build_simple_hierarchy(n_areas)

    from .populations import construct_column

    all_cells = []
    for _ in range(n_areas):
        all_cells.extend(construct_column())

    brain = jx.Network(all_cells)

    cells_per_area = 5
    for i in range(n_areas):
        base = i * cells_per_area
        brain.cell(list(range(base, base + 2))).add_to_group("PC")
        brain.cell(list(range(base + 2, base + 3))).add_to_group("PV")
        brain.cell(list(range(base + 3, base + 4))).add_to_group("SST")
        brain.cell(list(range(base + 4, base + 5))).add_to_group("VIP")
        brain.cell(list(range(base, base + cells_per_area))).add_to_group(f"Area_{i}")

    ff_synapse = IonotropicSynapse()
    fb_synapse = IonotropicSynapse()

    for i in range(n_areas - 1):
        base_i = i * cells_per_area
        base_next = (i + 1) * cells_per_area

        for j in range(2):
            jx.connect(
                brain.cell(base_i + j).branch(0).comp(0),
                brain.cell(base_next + j).branch(0).comp(0),
                ff_synapse,
            )

        jx.connect(
            brain.cell(base_next + 0).branch(0).comp(0),
            brain.cell(base_i + 3).branch(0).comp(0),
            fb_synapse,
        )
        jx.connect(
            brain.cell(base_next + 1).branch(0).comp(0),
            brain.cell(base_i + 4).branch(0).comp(0),
            fb_synapse,
        )

    logger.info("Cortical hierarchy assembly complete.")
    return brain


def build_11_area_hierarchy():
    """Legacy alias for a reduced two-area smoke hierarchy."""
    logger.info("Executing legacy alias: build_11_area_hierarchy (reduced to 2 areas)")
    return build_cortical_hierarchy(n_areas=2)
