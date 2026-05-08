"""Laminar three-area cortical oddball/omission tutorial model.

This module is a compact, reproducible bridge between the tutorial notebooks and the
``jbiophysic`` network/TFNE primitives.  It is intentionally lightweight: the model is a
scientific tutorial scaffold, not a validated biological simulation.

Design target
-------------
* three compact cortical areas: ``low``, ``mid``, ``high``;
* three laminar compartments: superficial, middle/L4-like, deep;
* four cell classes in ``[E, PV, SST, VIP]`` order, where Lichtenfeld et al. markers are
  mapped as ``Pyr -> E``, ``PV -> PV``, ``CB -> SST`` and ``CR -> VIP``;
* Izhikevich point-neuron dynamics in native mV/ms current-like units;
* receptor-specific synapse state variables for AMPA, NMDA and GABA;
* shared four-event timing for global-oddball and omission tasks.

The density priors are deliberately approximate three-bin summaries of the laminar trends
reported in Lichtenfeld et al. 2024.  They preserve the qualitative constraints most useful
for a tutorial: PV is strongest around the middle layer, CB/SST and CR/VIP are stronger in
superficial cortex, pyramidal cells peak in superficial/deep bins, PV/CB decrease with
hierarchical level, and CR/VIP increases with hierarchical level.  Replace these priors with
raw digitized/histology tables before making quantitative anatomical claims.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Literal

import numpy as np

from jbiophysic.networks.cortex import CortexNetworkSpec, make_cortex_network

AREA_ORDER: tuple[str, str, str] = ("low", "mid", "high")
LAYER_ORDER: tuple[str, str, str] = ("superficial", "middle", "deep")
CELL_ORDER: tuple[str, str, str, str] = ("E", "PV", "SST", "VIP")
RECEPTOR_ORDER: tuple[str, str, str] = ("AMPA", "GABA", "NMDA")


@dataclass(frozen=True)
class FourEventTiming:
    """Shared timing for global-oddball and omission tasks.

    Times are in milliseconds.  The default 500-ms grid uses a 500-ms baseline, four
    500-ms event slots, 500-ms inter-event gaps, and a 1000-ms post-sequence interval,
    producing exactly 5000 ms.  This intentionally harmonizes the global-oddball timing
    with omission timing so the model can compare conditions without changing the clock.
    """

    total_ms: float = 5000.0
    baseline_ms: float = 500.0
    event_ms: float = 500.0
    gap_ms: float = 500.0
    post_ms: float = 1000.0

    @property
    def slots(self) -> tuple[tuple[str, float, float], ...]:
        t = self.baseline_ms
        out: list[tuple[str, float, float]] = []
        for k in range(4):
            out.append((f"P{k + 1}", t, t + self.event_ms))
            t += self.event_ms
            if k < 3:
                out.append((f"D{k + 1}", t, t + self.gap_ms))
                t += self.gap_ms
        out.append(("post", t, self.total_ms))
        return tuple(out)

    @property
    def event_slots(self) -> tuple[tuple[str, float, float], ...]:
        return tuple(slot for slot in self.slots if slot[0].startswith("P"))


def lichtenfeld_three_layer_priors() -> dict[str, np.ndarray]:
    """Return tutorial laminar density priors in ``[E, PV, SST, VIP]`` order.

    Rows correspond to ``[superficial, middle, deep]``.  Values are percentages and each
    row sums to 100.  All entries are positive so small tutorial networks do not lose a
    cell class in a layer after integer rounding.
    """

    return {
        # Early/low sensory cortex: relatively stronger PV and CB/SST, lower CR/VIP.
        "low": np.asarray(
            [
                [68.0, 10.0, 12.0, 10.0],
                [68.0, 20.0, 7.0, 5.0],
                [78.0, 10.0, 7.0, 5.0],
            ],
            dtype=float,
        ),
        # Intermediate cortex: transitional density pattern.
        "mid": np.asarray(
            [
                [70.0, 9.0, 11.0, 10.0],
                [72.0, 16.0, 6.0, 6.0],
                [80.0, 8.0, 6.0, 6.0],
            ],
            dtype=float,
        ),
        # Higher cortex: lower PV/CB-SST, higher CR/VIP fraction.
        "high": np.asarray(
            [
                [68.0, 7.0, 10.0, 15.0],
                [72.0, 10.0, 6.0, 12.0],
                [78.0, 6.0, 6.0, 10.0],
            ],
            dtype=float,
        ),
    }


@dataclass(frozen=True)
class Event:
    """A single stimulus event within a trial."""
    name: str
    onset_ms: float
    duration_ms: float
    symbol: str
    drive_scale: float


@dataclass(frozen=True)
class TrialSchedule:
    """A full sequence of events for a simulation trial."""
    task: str
    condition: str
    events: tuple[Event, ...]
    total_ms: float


@dataclass(frozen=True)
class ThreeAreaCortexSpec:
    """Static specification for a compact three-area laminar cortex."""

    area_names: tuple[str, ...]
    layer_names: tuple[str, ...]
    cell_types: tuple[str, ...]
    positions_mm: np.ndarray
    positions_m: np.ndarray
    area_index: np.ndarray
    layer_index: np.ndarray
    cell_type_index: np.ndarray
    area_networks: tuple[CortexNetworkSpec, ...]
    edges: Mapping[str, np.ndarray]
    dt_ms: float
    duration_ms: float
    density_priors_percent: Mapping[str, np.ndarray]
    layer_fractions: tuple[float, float, float]
    timing: FourEventTiming

    @property
    def n_neurons(self) -> int:
        return int(self.positions_mm.shape[0])

    @property
    def cell_type_names(self) -> np.ndarray:
        names = np.asarray(self.cell_types, dtype=object)
        return names[self.cell_type_index]

    @property
    def layer_name_array(self) -> np.ndarray:
        names = np.asarray(self.layer_names, dtype=object)
        return names[self.layer_index]

    @property
    def area_name_array(self) -> np.ndarray:
        names = np.asarray(self.area_names, dtype=object)
        return names[self.area_index]

    def counts_by_area_layer_type(self) -> np.ndarray:
        out = np.zeros(
            (len(self.area_names), len(self.layer_names), len(self.cell_types)), dtype=int
        )
        for a, layer, ctype in zip(
            self.area_index, self.layer_index, self.cell_type_index, strict=True
        ):
            out[int(a), int(layer), int(ctype)] += 1
        return out


def _validate_density_priors(density_priors: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    for area in AREA_ORDER:
        if area not in density_priors:
            raise ValueError(f"missing density prior for area {area!r}")
        arr = np.asarray(density_priors[area], dtype=float)
        if arr.shape != (3, 4):
            raise ValueError("each density prior must have shape (3, 4)")
        if not np.all(np.isfinite(arr)):
            raise ValueError("density prior contains non-finite values")
        if np.any(arr <= 0.0):
            raise ValueError("density priors must be strictly positive for no-zero classes")
        row_sum = arr.sum(axis=1)
        if not np.allclose(row_sum, 100.0, atol=1e-6):
            raise ValueError("density prior rows must sum to 100")
        out[area] = arr
    return out


def _append_receptor_edge(
    sources: list[int],
    targets: list[int],
    receptor_index: list[int],
    weights: list[float],
    plasticity: list[float],
    connection_class: list[str],
    pre: int,
    post: int,
    receptor: Literal["AMPA", "GABA", "NMDA"],
    weight: float,
    plasticity_scale: float,
    klass: str,
) -> None:
    sources.append(int(pre))
    targets.append(int(post))
    receptor_index.append(RECEPTOR_ORDER.index(receptor))
    weights.append(float(weight))
    plasticity.append(float(plasticity_scale))
    connection_class.append(klass)


def _add_biophysical_edges(
    *,
    sources: list[int],
    targets: list[int],
    receptor_index: list[int],
    weights: list[float],
    plasticity: list[float],
    connection_class: list[str],
    pre: int,
    post: int,
    pre_type: str,
    klass: str,
    ampa_scale: float,
    nmda_scale: float,
    gaba_pv_scale: float,
    gaba_sst_vip_scale: float,
) -> None:
    if pre == post:
        return
    if pre_type == "E":
        _append_receptor_edge(
            sources,
            targets,
            receptor_index,
            weights,
            plasticity,
            connection_class,
            pre,
            post,
            "AMPA",
            0.045,
            ampa_scale,
            klass,
        )
        _append_receptor_edge(
            sources,
            targets,
            receptor_index,
            weights,
            plasticity,
            connection_class,
            pre,
            post,
            "NMDA",
            0.012,
            nmda_scale,
            klass,
        )
    else:
        g_scale = gaba_pv_scale if pre_type == "PV" else gaba_sst_vip_scale
        _append_receptor_edge(
            sources,
            targets,
            receptor_index,
            weights,
            plasticity,
            connection_class,
            pre,
            post,
            "GABA",
            0.070 if pre_type == "PV" else 0.035,
            g_scale,
            klass,
        )


def _make_edges(
    *,
    area_index: np.ndarray,
    layer_index: np.ndarray,
    cell_type_index: np.ndarray,
    rng: np.random.Generator,
    ampa_plasticity: float,
    nmda_plasticity: float,
    gaba_pv_plasticity: float,
    gaba_sst_vip_plasticity: float,
) -> dict[str, np.ndarray]:
    sources: list[int] = []
    targets: list[int] = []
    receptor_index: list[int] = []
    weights: list[float] = []
    plasticity: list[float] = []
    connection_class: list[str] = []

    type_names = np.asarray(CELL_ORDER, dtype=object)

    # Within-layer all-to-all inside each area/layer.
    for area in range(3):
        for layer in range(3):
            idx = np.flatnonzero((area_index == area) & (layer_index == layer))
            for pre in idx:
                pre_type = str(type_names[cell_type_index[pre]])
                for post in idx:
                    _add_biophysical_edges(
                        sources=sources,
                        targets=targets,
                        receptor_index=receptor_index,
                        weights=weights,
                        plasticity=plasticity,
                        connection_class=connection_class,
                        pre=int(pre),
                        post=int(post),
                        pre_type=pre_type,
                        klass="within_layer_all_to_all",
                        ampa_scale=ampa_plasticity,
                        nmda_scale=nmda_plasticity,
                        gaba_pv_scale=gaba_pv_plasticity,
                        gaba_sst_vip_scale=gaba_sst_vip_plasticity,
                    )

    # Between layers inside an area: type-to-type only.
    for area in range(3):
        for ctype in range(4):
            idx = np.flatnonzero((area_index == area) & (cell_type_index == ctype))
            for pre in idx:
                pre_type = str(type_names[cell_type_index[pre]])
                for post in idx:
                    if layer_index[pre] == layer_index[post]:
                        continue
                    _add_biophysical_edges(
                        sources=sources,
                        targets=targets,
                        receptor_index=receptor_index,
                        weights=weights,
                        plasticity=plasticity,
                        connection_class=connection_class,
                        pre=int(pre),
                        post=int(post),
                        pre_type=pre_type,
                        klass="between_layer_type_to_type",
                        ampa_scale=ampa_plasticity,
                        nmda_scale=nmda_plasticity,
                        gaba_pv_scale=gaba_pv_plasticity,
                        gaba_sst_vip_scale=gaba_sst_vip_plasticity,
                    )

    # Feedforward: superficial E of lower area -> middle/L4-like layer of next area.
    for lower, higher in [(0, 1), (1, 2)]:
        pre_idx = np.flatnonzero(
            (area_index == lower) & (layer_index == 0) & (cell_type_index == 0)
        )
        post_idx = np.flatnonzero((area_index == higher) & (layer_index == 1))
        for pre in pre_idx:
            for post in post_idx:
                _add_biophysical_edges(
                    sources=sources,
                    targets=targets,
                    receptor_index=receptor_index,
                    weights=weights,
                    plasticity=plasticity,
                    connection_class=connection_class,
                    pre=int(pre),
                    post=int(post),
                    pre_type="E",
                    klass="feedforward_superficial_to_middle",
                    ampa_scale=ampa_plasticity,
                    nmda_scale=nmda_plasticity,
                    gaba_pv_scale=gaba_pv_plasticity,
                    gaba_sst_vip_scale=gaba_sst_vip_plasticity,
                )

    # Feedback: superficial and deep E of higher area -> deep layer of lower area.
    for lower, higher in [(0, 1), (1, 2)]:
        pre_idx = np.flatnonzero(
            (area_index == higher)
            & np.isin(layer_index, [0, 2])
            & (cell_type_index == 0)
        )
        post_idx = np.flatnonzero((area_index == lower) & (layer_index == 2))
        for pre in pre_idx:
            for post in post_idx:
                _add_biophysical_edges(
                    sources=sources,
                    targets=targets,
                    receptor_index=receptor_index,
                    weights=weights,
                    plasticity=plasticity,
                    connection_class=connection_class,
                    pre=int(pre),
                    post=int(post),
                    pre_type="E",
                    klass="feedback_superficial_deep_to_deep",
                    ampa_scale=ampa_plasticity,
                    nmda_scale=nmda_plasticity,
                    gaba_pv_scale=gaba_pv_plasticity,
                    gaba_sst_vip_scale=gaba_sst_vip_plasticity,
                )

    # Apical inhibition: half of deep E cells receive SST inhibition.
    for area in range(3):
        deep_e = np.flatnonzero(
            (area_index == area) & (layer_index == 2) & (cell_type_index == 0)
        )
        if deep_e.size == 0:
            continue
        selected = rng.choice(deep_e, size=max(1, deep_e.size // 2), replace=False)
        sst_sources = np.flatnonzero((area_index == area) & (cell_type_index == 2))
        for pre in sst_sources:
            for post in selected:
                _append_receptor_edge(
                    sources,
                    targets,
                    receptor_index,
                    weights,
                    plasticity,
                    connection_class,
                    int(pre),
                    int(post),
                    "GABA",
                    0.045,
                    gaba_sst_vip_plasticity,
                    "apical_sst_to_half_deep_e",
                )

    return {
        "source": np.asarray(sources, dtype=np.int32),
        "target": np.asarray(targets, dtype=np.int32),
        "receptor_index": np.asarray(receptor_index, dtype=np.int8),
        "weight": np.asarray(weights, dtype=np.float32),
        "plasticity_scale": np.asarray(plasticity, dtype=np.float32),
        "receptor_names": np.asarray(RECEPTOR_ORDER, dtype=object),
        "connection_class": np.asarray(connection_class, dtype=object),
    }


def build_three_area_cortex(
    *,
    n_neurons: int = 300,
    xyz_per_area_mm: Sequence[float] = (0.5, 0.5, 1.5),
    layer_fractions: Sequence[float] = (0.45, 0.20, 0.35),
    density_priors: Mapping[str, np.ndarray] | None = None,
    dt_ms: float = 0.1,
    duration_ms: float = 5000.0,
    seed: int = 0,
    min_distance_um: float = 8.0,
    ampa_plasticity: float = 0.1,
    nmda_plasticity: float = 1.0,
    gaba_pv_plasticity: float = 1.0,
    gaba_sst_vip_plasticity: float = 0.1,
) -> ThreeAreaCortexSpec:
    """Build the tutorial three-area laminar cortex requested for oddball/omission.

    The default is 300 neurons with 100 neurons per area.  For notebook smoke tests, use
    a smaller ``n_neurons`` divisible by three, e.g. 60 or 90.
    """

    if n_neurons <= 0 or n_neurons % 3 != 0:
        raise ValueError("n_neurons must be positive and divisible by 3")
    if dt_ms <= 0 or duration_ms <= 0:
        raise ValueError("dt_ms and duration_ms must be positive")
    if len(layer_fractions) != 3:
        raise ValueError("layer_fractions must have three entries")

    density = _validate_density_priors(density_priors or lichtenfeld_three_layer_priors())
    n_per_area = n_neurons // 3
    xyz = np.asarray(xyz_per_area_mm, dtype=float)
    if xyz.shape != (3,) or np.any(xyz <= 0):
        raise ValueError("xyz_per_area_mm must have three positive entries")

    rng = np.random.default_rng(seed)
    area_specs: list[CortexNetworkSpec] = []
    positions_mm_list: list[np.ndarray] = []
    positions_m_list: list[np.ndarray] = []
    area_idx_list: list[np.ndarray] = []
    layer_idx_list: list[np.ndarray] = []
    type_idx_list: list[np.ndarray] = []
    x_gap = 0.15 * xyz[0]

    for ai, area_name in enumerate(AREA_ORDER):
        spec = make_cortex_network(
            XYZ_mm=xyz.tolist(),
            N=n_per_area,
            Ls=layer_fractions,
            Ld=density[area_name],
            model_family="izhikevich",
            plasticity_coefficient=0.0,
            seed=seed + 100 * ai,
            min_distance_um=min_distance_um,
            make_synapses=False,
        )
        offset_mm = np.asarray([ai * (xyz[0] + x_gap), 0.0, 0.0], dtype=float)
        pos_mm = spec.positions_mm + offset_mm
        area_specs.append(spec)
        positions_mm_list.append(pos_mm)
        positions_m_list.append(pos_mm * 1e-3)
        area_idx_list.append(np.full(spec.n_neurons, ai, dtype=np.int16))
        layer_idx_list.append(spec.layer_index.astype(np.int16))
        type_idx_list.append(spec.cell_type_index.astype(np.int16))

    positions_mm = np.vstack(positions_mm_list)
    positions_m = np.vstack(positions_m_list)
    area_index = np.concatenate(area_idx_list)
    layer_index = np.concatenate(layer_idx_list)
    cell_type_index = np.concatenate(type_idx_list)

    # Verify no area/layer/type was rounded to zero.
    tmp = np.zeros((3, 3, 4), dtype=int)
    for a, layer, ctype in zip(area_index, layer_index, cell_type_index, strict=True):
        tmp[int(a), int(layer), int(ctype)] += 1
    if np.any(tmp == 0):
        raise RuntimeError(
            "integer allocation produced a zero area/layer/type bin; increase n_neurons or adjust priors"
        )

    edges = _make_edges(
        area_index=area_index,
        layer_index=layer_index,
        cell_type_index=cell_type_index,
        rng=rng,
        ampa_plasticity=ampa_plasticity,
        nmda_plasticity=nmda_plasticity,
        gaba_pv_plasticity=gaba_pv_plasticity,
        gaba_sst_vip_plasticity=gaba_sst_vip_plasticity,
    )

    return ThreeAreaCortexSpec(
        area_names=AREA_ORDER,
        layer_names=LAYER_ORDER,
        cell_types=CELL_ORDER,
        positions_mm=positions_mm,
        positions_m=positions_m,
        area_index=area_index,
        layer_index=layer_index,
        cell_type_index=cell_type_index,
        area_networks=tuple(area_specs),
        edges=edges,
        dt_ms=float(dt_ms),
        duration_ms=float(duration_ms),
        density_priors_percent=density,
        layer_fractions=tuple(float(x) / float(np.sum(layer_fractions)) for x in layer_fractions),
        timing=FourEventTiming(total_ms=float(duration_ms)),
    )


def global_oddball_sequences() -> dict[str, str]:
    """Canonical global/local oddball sequences using the same four event slots."""

    return {
        "habituated_local_oddball": "AAAB",
        "global_oddball": "AAAA",
        "control_A": "AAAA",
        "control_B": "BBBB",
    }


def omission_sequences() -> dict[str, str]:
    """Omission sequences sharing the same four event slots; X means no input."""

    return {
        "standard_A": "AAAB",
        "omit_P2_A": "AXAB",
        "omit_P3_A": "AAXB",
        "omit_P4_A": "AAAX",
        "standard_B": "BBBA",
        "omit_P2_B": "BXBA",
        "omit_P3_B": "BBXA",
        "omit_P4_B": "BBBX",
        "random_R": "RRRR",
        "omit_P2_R": "RXRR",
        "omit_P3_R": "RRXR",
        "omit_P4_R": "RRRX",
    }


def drive_schedule(
    cortex: ThreeAreaCortexSpec,
    sequence: str,
    *,
    amplitude: float = 6.0,
    background: float = 2.0,
    feature_mode: Literal["all", "split"] = "split",
) -> np.ndarray:
    """Return a `[time, neuron]` drive matrix in Izhikevich current-like units.

    Stimulus drive enters the middle/L4-like layer of the lower area.  In ``split`` mode,
    A and B drive opposite halves of lower-area middle-layer E cells, while R drives all of
    them.  X is omission and injects no event drive.
    """

    if len(sequence) != 4:
        raise ValueError("sequence must have four symbols")
    dt_ms = cortex.dt_ms
    steps = int(round(cortex.duration_ms / dt_ms))
    drive = np.full((steps, cortex.n_neurons), float(background), dtype=np.float32)

    target = np.flatnonzero(
        (cortex.area_index == 0) & (cortex.layer_index == 1) & (cortex.cell_type_index == 0)
    )
    if target.size == 0:
        raise RuntimeError("lower middle E target set is empty")
    half = max(1, target.size // 2)
    group_A = target[:half]
    group_B = target[half:] if target.size > 1 else target

    for symbol, (_slot_name, t0, t1) in zip(sequence, cortex.timing.event_slots, strict=True):
        i0 = max(0, int(round(t0 / dt_ms)))
        i1 = min(steps, int(round(t1 / dt_ms)))
        if symbol == "X":
            continue
        if feature_mode == "all" or symbol == "R":
            idx = target
        elif symbol == "A":
            idx = group_A
        elif symbol == "B":
            idx = group_B
        else:
            raise ValueError(f"unknown sequence symbol {symbol!r}")
        drive[i0:i1, idx] += float(amplitude)
    return drive



def density_priors_table(density_priors: Mapping[str, np.ndarray] | None = None) -> list[dict[str, float | str]]:
    """Return density priors as a tidy table-friendly list of dictionaries.

    The values are percentages in the tutorial's three-layer abstraction.  They are
    Lichtenfeld-inspired priors, not digitized raw histology.  The function exists so
    notebooks, tests, and downstream users can inspect the anatomical assumptions in a
    machine-readable form instead of copying hidden constants.
    """

    density = _validate_density_priors(density_priors or lichtenfeld_three_layer_priors())
    rows: list[dict[str, float | str]] = []
    for area in AREA_ORDER:
        arr = density[area]
        for li, layer in enumerate(LAYER_ORDER):
            row: dict[str, float | str] = {"area": area, "layer": layer}
            for ci, ctype in enumerate(CELL_ORDER):
                row[ctype] = float(arr[li, ci])
            rows.append(row)
    return rows


def default_plasticity_scales() -> dict[str, float]:
    """Return the receptor/cell-class plasticity scales requested for the tutorial."""

    return {
        "AMPA_all": 0.1,
        "NMDA_all": 1.0,
        "GABA_PV": 1.0,
        "GABA_SST_VIP": 0.1,
    }


def validate_replication_constraints(cortex: ThreeAreaCortexSpec) -> dict[str, bool | int | float]:
    """Validate the compact replication scaffold against the requested constraints.

    This is intentionally explicit and redundant.  It gives tutorials and worker agents a
    single audit hook before interpreting any simulated response.
    """

    counts = cortex.counts_by_area_layer_type()
    edge = cortex.edges
    src = np.asarray(edge["source"], dtype=int)
    tgt = np.asarray(edge["target"], dtype=int)
    receptor = np.asarray(edge["receptor_index"], dtype=int)
    klass = np.asarray(edge["connection_class"], dtype=object)
    receptor_names = np.asarray(edge["receptor_names"], dtype=object)
    source_types = cortex.cell_type_names[src]
    interarea = cortex.area_index[src] != cortex.area_index[tgt]
    interarea_receptors = set(receptor_names[receptor[interarea]].tolist()) if np.any(interarea) else set()
    deep_e = np.flatnonzero((cortex.layer_index == 2) & (cortex.cell_type_index == 0))
    apical_targets = np.unique(tgt[klass == "apical_sst_to_half_deep_e"])

    return {
        "n_neurons": int(cortex.n_neurons),
        "three_areas": len(cortex.area_names) == 3,
        "three_layers": len(cortex.layer_names) == 3,
        "four_cell_types": tuple(cortex.cell_types) == CELL_ORDER,
        "no_zero_area_layer_type_bins": bool(np.all(counts > 0)),
        "equal_area_counts": bool(np.all(counts.sum(axis=(1, 2)) == counts.sum(axis=(1, 2))[0])),
        "layer_counts_match_45_20_35_per_area": bool(
            np.all(counts.sum(axis=2) == np.asarray([45, 20, 35]))
        ) if cortex.n_neurons == 300 else bool(np.all(counts.sum(axis=2) > 0)),
        "e_sources_are_excitatory_receptors": bool(
            np.all(np.isin(receptor_names[receptor[source_types == "E"]], ["AMPA", "NMDA"]))
        ),
        "inhibitory_sources_are_gaba": bool(
            np.all(receptor_names[receptor[source_types != "E"]] == "GABA")
        ),
        "interarea_connections_are_excitatory": interarea_receptors.issubset({"AMPA", "NMDA"}),
        "has_feedforward_path": bool(np.any(klass == "feedforward_superficial_to_middle")),
        "has_feedback_path": bool(np.any(klass == "feedback_superficial_deep_to_deep")),
        "has_apical_sst_to_deep_e": bool(np.any(klass == "apical_sst_to_half_deep_e")),
        "apical_targets_fraction_of_deep_e": float(len(np.intersect1d(apical_targets, deep_e)) / max(1, deep_e.size)),
        "duration_ms": float(cortex.duration_ms),
        "dt_ms": float(cortex.dt_ms),
    }


def edge_mask(
    cortex: ThreeAreaCortexSpec,
    *,
    receptor: str | None = None,
    connection_class: str | None = None,
    source_type: str | None = None,
    target_type: str | None = None,
    source_area: str | None = None,
    target_area: str | None = None,
    source_layer: str | None = None,
    target_layer: str | None = None,
) -> np.ndarray:
    """Return a Boolean edge mask for receptor-, class-, type-, area-, or layer-specific edits."""

    src = np.asarray(cortex.edges["source"], dtype=int)
    tgt = np.asarray(cortex.edges["target"], dtype=int)
    mask = np.ones(src.shape[0], dtype=bool)
    receptor_names = np.asarray(cortex.edges["receptor_names"], dtype=object)
    if receptor is not None:
        mask &= receptor_names[np.asarray(cortex.edges["receptor_index"], dtype=int)] == receptor
    if connection_class is not None:
        mask &= np.asarray(cortex.edges["connection_class"], dtype=object) == connection_class
    if source_type is not None:
        mask &= cortex.cell_type_names[src] == source_type
    if target_type is not None:
        mask &= cortex.cell_type_names[tgt] == target_type
    if source_area is not None:
        mask &= cortex.area_name_array[src] == source_area
    if target_area is not None:
        mask &= cortex.area_name_array[tgt] == target_area
    if source_layer is not None:
        mask &= cortex.layer_name_array[src] == source_layer
    if target_layer is not None:
        mask &= cortex.layer_name_array[tgt] == target_layer
    return mask


def perturb_cortex_edges(
    cortex: ThreeAreaCortexSpec,
    *,
    scale: float = 1.0,
    receptor: str | None = None,
    connection_class: str | None = None,
    source_type: str | None = None,
    target_type: str | None = None,
    source_area: str | None = None,
    target_area: str | None = None,
    source_layer: str | None = None,
    target_layer: str | None = None,
) -> ThreeAreaCortexSpec:
    """Return a copy of ``cortex`` with selected edge weights scaled.

    Examples
    --------
    * PV silencing: ``source_type="PV", receptor="GABA", scale=0``.
    * remove feedback: ``connection_class="feedback_superficial_deep_to_deep", scale=0``.
    * strengthen NMDA: ``receptor="NMDA", scale=1.5``.
    """

    if not np.isfinite(scale) or scale < 0:
        raise ValueError("scale must be finite and nonnegative")
    mask = edge_mask(
        cortex,
        receptor=receptor,
        connection_class=connection_class,
        source_type=source_type,
        target_type=target_type,
        source_area=source_area,
        target_area=target_area,
        source_layer=source_layer,
        target_layer=target_layer,
    )
    new_edges = {k: np.asarray(v).copy() for k, v in cortex.edges.items()}
    new_edges["weight"] = new_edges["weight"].astype(np.float32, copy=True)
    new_edges["weight"][mask] *= float(scale)
    return replace(cortex, edges=new_edges)


def task_sequence_catalog() -> dict[str, dict[str, str]]:
    """Return all tutorial task conditions using the same four event slots."""

    return {
        "global_oddball": global_oddball_sequences(),
        "omission": omission_sequences(),
    }


def build_drive_batch(
    cortex: ThreeAreaCortexSpec,
    sequences: Mapping[str, str],
    *,
    amplitude: float = 6.0,
    background: float = 2.0,
    feature_mode: Literal["all", "split"] = "split",
) -> dict[str, np.ndarray]:
    """Build drive matrices for a named sequence dictionary."""

    return {
        name: drive_schedule(
            cortex,
            seq,
            amplitude=amplitude,
            background=background,
            feature_mode=feature_mode,
        )
        for name, seq in sequences.items()
    }


def simulate_condition_batch(
    cortex: ThreeAreaCortexSpec,
    drives: Mapping[str, np.ndarray],
    *,
    seed: int = 0,
    noise_sd: float = 0.5,
    plasticity_enabled: bool = True,
) -> dict[str, dict[str, np.ndarray]]:
    """Simulate several named conditions with deterministic per-condition seeds."""

    out: dict[str, dict[str, np.ndarray]] = {}
    for i, (name, drive) in enumerate(drives.items()):
        out[name] = simulate_laminar_izhikevich(
            cortex,
            drive,
            seed=seed + i,
            noise_sd=noise_sd,
            plasticity_enabled=plasticity_enabled,
        )
    return out


def _slot_slice(cortex: ThreeAreaCortexSpec, slot_index: int) -> slice:
    _name, t0, t1 = cortex.timing.event_slots[slot_index]
    return slice(int(round(t0 / cortex.dt_ms)), int(round(t1 / cortex.dt_ms)))


def firing_rate_by_group(
    result: Mapping[str, np.ndarray],
    cortex: ThreeAreaCortexSpec,
    *,
    group_by: Literal["area", "layer", "cell_type"] = "area",
    window: slice | None = None,
) -> dict[str, float]:
    """Return mean firing-rate by area, layer, or cell type for a window."""

    spikes = np.asarray(result["spikes"], dtype=bool)
    if window is not None:
        spikes = spikes[window]
    duration_s = spikes.shape[0] * cortex.dt_ms / 1000.0
    if group_by == "area":
        labels = cortex.area_name_array
    elif group_by == "layer":
        labels = cortex.layer_name_array
    elif group_by == "cell_type":
        labels = cortex.cell_type_names
    else:  # pragma: no cover
        raise ValueError("unknown group_by")
    out: dict[str, float] = {}
    for label in np.unique(labels):
        idx = labels == label
        if duration_s <= 0.0 or spikes.shape[0] == 0:
            out[str(label)] = 0.0
        else:
            out[str(label)] = float(spikes[:, idx].sum() / max(1, idx.sum()) / duration_s)
    return out


def sequence_p4_minus_p3_contrast(result: Mapping[str, np.ndarray], cortex: ThreeAreaCortexSpec) -> float:
    """Return population firing contrast P4 - P3 in Hz."""

    rates_p3 = firing_rate_by_group(result, cortex, group_by="area", window=_slot_slice(cortex, 2))
    rates_p4 = firing_rate_by_group(result, cortex, group_by="area", window=_slot_slice(cortex, 3))
    return float(np.mean(list(rates_p4.values())) - np.mean(list(rates_p3.values())))


def oddball_objectives(
    results: Mapping[str, Mapping[str, np.ndarray]],
    cortex: ThreeAreaCortexSpec,
) -> dict[str, float]:
    """Compute compact tutorial objectives for local/global oddball response patterns.

    The values are not inferential statistics.  They are optimization targets/checks for
    whether a toy model expresses the qualitative response motifs.
    """

    out: dict[str, float] = {}
    if "habituated_local_oddball" in results and "control_A" in results:
        out["local_oddball_p4_minus_p3_excess_hz"] = (
            sequence_p4_minus_p3_contrast(results["habituated_local_oddball"], cortex)
            - sequence_p4_minus_p3_contrast(results["control_A"], cortex)
        )
    if "global_oddball" in results and "control_A" in results:
        global_p4 = firing_rate_by_group(results["global_oddball"], cortex, window=_slot_slice(cortex, 3))
        control_p4 = firing_rate_by_group(results["control_A"], cortex, window=_slot_slice(cortex, 3))
        out["global_high_minus_low_p4_excess_hz"] = float(
            (global_p4.get("high", 0.0) - control_p4.get("high", 0.0))
            - (global_p4.get("low", 0.0) - control_p4.get("low", 0.0))
        )
    return out


def omission_objectives(
    results: Mapping[str, Mapping[str, np.ndarray]],
    cortex: ThreeAreaCortexSpec,
    *,
    standard_name: str = "standard_A",
    omission_name: str = "omit_P4_A",
) -> dict[str, float]:
    """Compute compact objectives for sparse higher-order omission modulation."""

    if standard_name not in results or omission_name not in results:
        return {}
    p4 = _slot_slice(cortex, 3)
    standard = firing_rate_by_group(results[standard_name], cortex, group_by="area", window=p4)
    omission = firing_rate_by_group(results[omission_name], cortex, group_by="area", window=p4)
    area_delta = {area: omission[area] - standard[area] for area in standard}
    return {
        "omission_high_minus_low_delta_hz": float(area_delta.get("high", 0.0) - area_delta.get("low", 0.0)),
        "omission_mean_delta_hz": float(np.mean(list(area_delta.values()))),
    }


def population_activity_proxy(
    result: Mapping[str, np.ndarray],
    cortex: ThreeAreaCortexSpec,
    *,
    group_by: Literal["area", "layer", "cell_type"] = "area",
    smoothing_ms: float = 5.0,
) -> dict[str, np.ndarray]:
    """Return smoothed spike-rate proxy traces by group.

    This is a lightweight proxy for MUA/LFP-like summaries in tutorials.  It is not a CSD or
    extracellular forward solution.
    """

    spikes = np.asarray(result["spikes"], dtype=float)
    width = max(1, int(round(smoothing_ms / cortex.dt_ms)))
    kernel = np.ones(width, dtype=float) / width
    if group_by == "area":
        labels = cortex.area_name_array
    elif group_by == "layer":
        labels = cortex.layer_name_array
    elif group_by == "cell_type":
        labels = cortex.cell_type_names
    else:  # pragma: no cover
        raise ValueError("unknown group_by")
    out: dict[str, np.ndarray] = {}
    for label in np.unique(labels):
        idx = labels == label
        raw = spikes[:, idx].mean(axis=1) * (1000.0 / cortex.dt_ms)
        out[str(label)] = np.convolve(raw, kernel, mode="same").astype(np.float32)
    return out


def tfne_source_proxy(
    result: Mapping[str, np.ndarray],
    cortex: ThreeAreaCortexSpec,
    *,
    spike_current_A: float = 1e-12,
) -> dict[str, np.ndarray]:
    """Return TFNE-ready sparse source positions and time-varying currents.

    Each spike contributes ``spike_current_A`` at the neuron's position for that time bin.
    This is only a calibrated bridge object; a proper TFNE readout must still project these
    sparse currents through normalized source kernels and solve/diagnose the field model.
    """

    if not np.isfinite(spike_current_A) or spike_current_A <= 0:
        raise ValueError("spike_current_A must be positive and finite")
    spikes = np.asarray(result["spikes"], dtype=np.float32)
    return {
        "positions_m": np.asarray(cortex.positions_m, dtype=np.float64),
        "currents_A": spikes * float(spike_current_A),
        "current_scale_A_per_spike": np.asarray(float(spike_current_A)),
        "truth_status": np.asarray("tfne_source_proxy_not_field_solution", dtype=object),
    }


def replication_manifest(cortex: ThreeAreaCortexSpec) -> dict[str, object]:
    """Return a JSON-serializable model/task manifest for tutorial provenance."""

    return {
        "model": "three_area_laminar_izhikevich_oddball_omission_scaffold",
        "truth_status": "tutorial_exploratory_not_biological_truth",
        "areas": list(cortex.area_names),
        "layers": list(cortex.layer_names),
        "cell_types": list(cortex.cell_types),
        "n_neurons": cortex.n_neurons,
        "dt_ms": cortex.dt_ms,
        "duration_ms": cortex.duration_ms,
        "layer_fractions": list(cortex.layer_fractions),
        "density_priors_percent": density_priors_table(cortex.density_priors_percent),
        "plasticity_scales": default_plasticity_scales(),
        "tasks": task_sequence_catalog(),
        "constraints": validate_replication_constraints(cortex),
    }


def izhikevich_params_for_types(cell_type_index: np.ndarray) -> tuple[np.ndarray, ...]:
    """Return vectorized Izhikevich parameters for E/PV/SST/VIP cells."""

    cti = np.asarray(cell_type_index)
    a = np.where(cti == 1, 0.10, np.where(cti == 2, 0.02, 0.02)).astype(np.float32)
    b = np.where(cti == 2, 0.25, 0.20).astype(np.float32)
    c = np.where(cti == 0, -65.0, np.where(cti == 3, -60.0, -65.0)).astype(np.float32)
    d = np.where(cti == 0, 8.0, np.where(cti == 1, 2.0, np.where(cti == 2, 2.0, 4.0))).astype(
        np.float32
    )
    return a, b, c, d


def simulate_laminar_izhikevich(
    cortex: ThreeAreaCortexSpec,
    drive: np.ndarray,
    *,
    seed: int = 0,
    noise_sd: float = 0.5,
    plasticity_enabled: bool = True,
    plasticity_lr: float = 1e-4,
    weight_max: float = 0.25,
) -> dict[str, np.ndarray]:
    """Run a lightweight edge-list Izhikevich simulation.

    This routine is optimized for transparent tutorial behavior and smoke tests.  It can run
    the full 300-neuron, 5000-ms configuration, but notebooks should default to shorter
    smoke runs unless the user explicitly opts in.
    """

    drive = np.asarray(drive, dtype=np.float32)
    steps, n = drive.shape
    if n != cortex.n_neurons:
        raise ValueError("drive second dimension must equal cortex.n_neurons")

    rng = np.random.default_rng(seed)
    a, b, c, d = izhikevich_params_for_types(cortex.cell_type_index)
    v = (-65.0 + 5.0 * rng.standard_normal(n)).astype(np.float32)
    u = (b * v).astype(np.float32)

    source = np.asarray(cortex.edges["source"], dtype=np.int32)
    target = np.asarray(cortex.edges["target"], dtype=np.int32)
    receptor = np.asarray(cortex.edges["receptor_index"], dtype=np.int8)
    weights = np.asarray(cortex.edges["weight"], dtype=np.float32).copy()
    plasticity = np.asarray(cortex.edges["plasticity_scale"], dtype=np.float32)
    sign = np.where(receptor == RECEPTOR_ORDER.index("GABA"), -1.0, 1.0).astype(np.float32)
    tau = np.where(
        receptor == RECEPTOR_ORDER.index("AMPA"),
        2.0,
        np.where(receptor == RECEPTOR_ORDER.index("GABA"), 10.0, 100.0),
    ).astype(np.float32)
    syn_state = np.zeros(source.shape[0], dtype=np.float32)
    decay = np.exp(-cortex.dt_ms / tau).astype(np.float32)

    V = np.zeros((steps, n), dtype=np.float32)
    spikes = np.zeros((steps, n), dtype=np.bool_)
    mean_rate_proxy = np.zeros(steps, dtype=np.float32)

    for t in range(steps):
        I_syn = np.zeros(n, dtype=np.float32)
        np.add.at(I_syn, target, sign * weights * syn_state)
        I = drive[t] + I_syn + noise_sd * rng.standard_normal(n).astype(np.float32)

        dv = 0.04 * v * v + 5.0 * v + 140.0 - u + I
        du = a * (b * v - u)
        v_pre = v + cortex.dt_ms * dv
        u_pre = u + cortex.dt_ms * du
        spiked = v_pre >= 30.0
        v = np.where(spiked, c, v_pre).astype(np.float32)
        u = np.where(spiked, u_pre + d, u_pre).astype(np.float32)

        syn_state = syn_state * decay + spiked[source].astype(np.float32)
        if plasticity_enabled and np.any(spiked):
            hebb = spiked[source].astype(np.float32) * spiked[target].astype(np.float32)
            weights += plasticity_lr * plasticity * hebb * (weight_max - weights)
            weights = np.clip(weights, 0.0, weight_max)

        V[t] = v
        spikes[t] = spiked
        mean_rate_proxy[t] = float(np.mean(spiked)) * (1000.0 / cortex.dt_ms)

    duration_s = (steps * cortex.dt_ms) / 1000.0
    firing_rate_hz = spikes.sum(axis=0).astype(np.float32) / duration_s
    return {
        "V_mV": V,
        "spikes": spikes,
        "mean_rate_proxy_hz": mean_rate_proxy,
        "firing_rate_hz": firing_rate_hz,
        "final_weights": weights,
        "finite": np.asarray(np.isfinite(V).all() and np.isfinite(weights).all()),
    }


def summarize_simulation(result: Mapping[str, np.ndarray], dt_ms: float) -> dict[str, float | bool]:
    """Return compact validation metrics for a simulation output."""

    V = np.asarray(result["V_mV"])
    spikes = np.asarray(result["spikes"])
    duration_s = spikes.shape[0] * dt_ms / 1000.0
    rates = spikes.sum(axis=0) / duration_s
    return {
        "finite": bool(np.isfinite(V).all() and np.isfinite(rates).all()),
        "n_spikes": int(spikes.sum()),
        "mean_rate_hz": float(np.mean(rates)),
        "max_rate_hz": float(np.max(rates)),
        "v_min_mV": float(np.min(V)),
        "v_max_mV": float(np.max(V)),
        "active_fraction": float(np.mean(rates > 0.0)),
    }
