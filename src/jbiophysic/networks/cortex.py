"""Layered 3-D cortical volume network builder.

This module creates a geometry-first network specification for jbiophysic.  It is
intentionally simulator-agnostic: the same generated positions, layer labels, cell-type
labels, and synapse metadata can be used by an Izhikevich point-neuron model, a TFNE
forward-field model, or a source-coupled TFNE-Izhikevich hybrid.

Coordinate convention
---------------------
Input ``XYZ_mm`` is the physical cortical volume in millimetres.  Layers span the z-axis.
Returned positions are in millimetres; TFNE helper fields that require SI units are also
provided in metres.

Cell-type convention
--------------------
The canonical row order for layer density rows is ``[E, PV, SST, VIP]``.  The default
projection rule is source-cell based: E sources use AMPA and inhibitory sources use GABA.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np

ModelFamily = Literal["izhikevich", "tfne", "tfne-izhikevich"]
CELL_ORDER: tuple[str, str, str, str] = ("E", "PV", "SST", "VIP")


@dataclass(frozen=True)
class CellTypeSpec:
    """Static metadata for a cortical cell class."""

    name: str
    waveform: str
    transmitter: Literal["AMPA", "GABA"]
    polarity: Literal["excitatory", "inhibitory"]
    izhikevich: Mapping[str, float]


DEFAULT_CELL_TYPES: Mapping[str, CellTypeSpec] = {
    "E": CellTypeSpec(
        name="E",
        waveform="regular_spiking",
        transmitter="AMPA",
        polarity="excitatory",
        izhikevich={"a": 0.02, "b": 0.2, "c": -65.0, "d": 8.0, "v_spike_mV": 30.0},
    ),
    "PV": CellTypeSpec(
        name="PV",
        waveform="fast_spiking",
        transmitter="GABA",
        polarity="inhibitory",
        izhikevich={"a": 0.1, "b": 0.2, "c": -65.0, "d": 2.0, "v_spike_mV": 30.0},
    ),
    "SST": CellTypeSpec(
        name="SST",
        waveform="low_threshold_spiking",
        transmitter="GABA",
        polarity="inhibitory",
        izhikevich={"a": 0.02, "b": 0.25, "c": -65.0, "d": 2.0, "v_spike_mV": 30.0},
    ),
    "VIP": CellTypeSpec(
        name="VIP",
        waveform="regular_spiking_disinhibitory",
        transmitter="GABA",
        polarity="inhibitory",
        izhikevich={"a": 0.02, "b": 0.2, "c": -65.0, "d": 8.0, "v_spike_mV": 30.0},
    ),
}


@dataclass(frozen=True)
class CortexNetworkSpec:
    """Complete static network specification produced by ``make_cortex_network``."""

    model_family: ModelFamily
    xyz_mm: tuple[float, float, float]
    layer_size_fraction: tuple[float, ...]
    layer_z_bounds_mm: tuple[tuple[float, float], ...]
    layer_density_fraction: tuple[tuple[float, float, float, float], ...]
    plasticity_coefficient: float
    min_distance_mm: float
    seed: int
    positions_mm: np.ndarray
    positions_m: np.ndarray
    layer_index: np.ndarray
    cell_type_index: np.ndarray
    cell_types: tuple[str, ...]
    cell_type_specs: Mapping[str, CellTypeSpec] = field(default_factory=lambda: DEFAULT_CELL_TYPES)
    synapses: Mapping[str, np.ndarray] = field(default_factory=dict)
    tfne: Mapping[str, Any] = field(default_factory=dict)

    @property
    def n_neurons(self) -> int:
        return int(self.positions_mm.shape[0])

    @property
    def cell_type_names(self) -> np.ndarray:
        names = np.asarray(self.cell_types, dtype=object)
        return names[self.cell_type_index]

    @property
    def layer_names(self) -> np.ndarray:
        return np.asarray([f"L{i + 1}" for i in range(len(self.layer_size_fraction))], dtype=object)[
            self.layer_index
        ]

    def counts_by_layer_and_type(self) -> np.ndarray:
        """Return integer count matrix with shape ``[n_layers, 4]``."""
        out = np.zeros((len(self.layer_size_fraction), len(self.cell_types)), dtype=int)
        for layer, ctype in zip(self.layer_index, self.cell_type_index, strict=True):
            out[int(layer), int(ctype)] += 1
        return out

    def to_json_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-safe dictionary."""
        synapses = {k: np.asarray(v).tolist() for k, v in self.synapses.items()}
        tfne = {
            k: (np.asarray(v).tolist() if isinstance(v, np.ndarray) else v)
            for k, v in self.tfne.items()
        }
        return {
            "model_family": self.model_family,
            "xyz_mm": list(self.xyz_mm),
            "layer_size_fraction": list(self.layer_size_fraction),
            "layer_z_bounds_mm": [list(x) for x in self.layer_z_bounds_mm],
            "layer_density_fraction": [list(x) for x in self.layer_density_fraction],
            "plasticity_coefficient": self.plasticity_coefficient,
            "min_distance_mm": self.min_distance_mm,
            "seed": self.seed,
            "positions_mm": self.positions_mm.tolist(),
            "positions_m": self.positions_m.tolist(),
            "layer_index": self.layer_index.tolist(),
            "cell_type_index": self.cell_type_index.tolist(),
            "cell_types": list(self.cell_types),
            "cell_type_specs": {name: asdict(spec) for name, spec in self.cell_type_specs.items()},
            "synapses": synapses,
            "tfne": tfne,
            "counts_by_layer_and_type": self.counts_by_layer_and_type().tolist(),
        }


def _as_float_array(name: str, value: Sequence[float], expected_len: int | None = None) -> np.ndarray:
    arr = np.asarray(value, dtype=float)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if expected_len is not None and arr.shape[0] != expected_len:
        raise ValueError(f"{name} must have length {expected_len}")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} contains non-finite values")
    return arr


def _normalize_simplex(name: str, value: Sequence[float], *, atol: float = 1e-6) -> np.ndarray:
    arr = _as_float_array(name, value)
    if np.any(arr < 0.0):
        raise ValueError(f"{name} entries must be nonnegative")
    total = float(np.sum(arr))
    if total <= 0.0:
        raise ValueError(f"{name} must have positive sum")
    arr = arr / total
    if not np.isclose(float(np.sum(arr)), 1.0, atol=atol):
        raise ValueError(f"{name} could not be normalized")
    return arr


def _normalize_layer_density(Ld: Sequence[Sequence[float]], n_layers: int) -> np.ndarray:
    arr = np.asarray(Ld, dtype=float)
    if arr.shape != (n_layers, len(CELL_ORDER)):
        raise ValueError(f"Ld must have shape ({n_layers}, {len(CELL_ORDER)}) in [E, PV, SST, VIP] order")
    if not np.all(np.isfinite(arr)):
        raise ValueError("Ld contains non-finite values")
    if np.any(arr < 0.0):
        raise ValueError("Ld entries must be nonnegative")
    row_sum = np.sum(arr, axis=1)
    if np.any(row_sum <= 0.0):
        raise ValueError("each Ld row must have positive sum")
    # Accept both percentages summing to 100 and fractions summing to 1, then normalize.
    return arr / row_sum[:, None]


def _largest_remainder_counts(total: int, fractions: np.ndarray) -> np.ndarray:
    if total < 0:
        raise ValueError("total must be nonnegative")
    if total == 0:
        return np.zeros(fractions.shape[0], dtype=int)
    raw = fractions * total
    base = np.floor(raw).astype(int)
    remainder = int(total - np.sum(base))
    if remainder > 0:
        order = np.argsort(-(raw - base))
        base[order[:remainder]] += 1
    return base


def _allocate_counts(N: int, Ls: np.ndarray, Ld_frac: np.ndarray) -> np.ndarray:
    layer_counts = _largest_remainder_counts(N, Ls)
    counts = np.vstack([_largest_remainder_counts(int(n), row) for n, row in zip(layer_counts, Ld_frac, strict=True)])
    correction = N - int(np.sum(counts))
    if correction != 0:
        counts[-1, int(np.argmax(Ld_frac[-1]))] += correction
    if int(np.sum(counts)) != N:
        raise RuntimeError("internal allocation error")
    return counts.astype(int)


def _iter_neighbor_bins(bin_key: tuple[int, int, int]) -> Iterable[tuple[int, int, int]]:
    i, j, k = bin_key
    for di in (-1, 0, 1):
        for dj in (-1, 0, 1):
            for dk in (-1, 0, 1):
                yield (i + di, j + dj, k + dk)


def _sample_non_overlapping_positions(
    *,
    rng: np.random.Generator,
    xyz_mm: np.ndarray,
    layer_z_bounds_mm: Sequence[tuple[float, float]],
    counts: np.ndarray,
    min_distance_mm: float,
    max_attempts_per_neuron: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Rejection sample positions with a cell-list acceleration structure."""
    if min_distance_mm < 0.0:
        raise ValueError("min_distance_mm must be nonnegative")
    if max_attempts_per_neuron < 1:
        raise ValueError("max_attempts_per_neuron must be positive")

    positions: list[np.ndarray] = []
    layers: list[int] = []
    ctypes: list[int] = []
    bins: dict[tuple[int, int, int], list[int]] = {}
    bin_width = max(min_distance_mm, 1e-12)
    min_d2 = min_distance_mm * min_distance_mm
    margin = 0.5 * min_distance_mm

    def bin_for(pos: np.ndarray) -> tuple[int, int, int]:
        return tuple(np.floor(pos / bin_width).astype(int).tolist())

    def is_valid(pos: np.ndarray) -> bool:
        if min_distance_mm == 0.0:
            return True
        for nb in _iter_neighbor_bins(bin_for(pos)):
            for idx in bins.get(nb, []):
                if float(np.sum((positions[idx] - pos) ** 2)) < min_d2:
                    return False
        return True

    for layer_idx, z_bounds in enumerate(layer_z_bounds_mm):
        z0, z1 = z_bounds
        lo = np.array([margin, margin, z0 + margin], dtype=float)
        hi = np.array([xyz_mm[0] - margin, xyz_mm[1] - margin, z1 - margin], dtype=float)
        if np.any(hi <= lo):
            raise ValueError(
                "min_distance_mm is too large for the requested volume/layer; "
                f"layer={layer_idx + 1}, lo={lo.tolist()}, hi={hi.tolist()}"
            )
        for ctype_idx, n in enumerate(counts[layer_idx]):
            for _ in range(int(n)):
                accepted = False
                for _attempt in range(max_attempts_per_neuron):
                    pos = rng.uniform(lo, hi)
                    if is_valid(pos):
                        idx = len(positions)
                        positions.append(pos)
                        layers.append(layer_idx)
                        ctypes.append(ctype_idx)
                        bins.setdefault(bin_for(pos), []).append(idx)
                        accepted = True
                        break
                if not accepted:
                    raise RuntimeError(
                        "could not place all neurons without overlap; reduce N, reduce "
                        "min_distance_um, increase XYZ_mm, or raise max_attempts_per_neuron"
                    )

    return np.asarray(positions, dtype=float), np.asarray(layers, dtype=int), np.asarray(ctypes, dtype=int)


def _default_connection_probability(source_type: str, target_type: str) -> float:
    if source_type == "E" and target_type == "E":
        return 0.10
    if source_type == "E":
        return 0.20
    if source_type == "PV":
        return 0.25
    if source_type == "SST":
        return 0.15
    if source_type == "VIP":
        return 0.20 if target_type in {"PV", "SST"} else 0.05
    return 0.10


def make_distance_synapses(
    positions_mm: np.ndarray,
    cell_type_index: np.ndarray,
    *,
    rng: np.random.Generator,
    cell_types: Sequence[str] = CELL_ORDER,
    plasticity_coefficient: float = 0.0,
    length_constant_mm: float = 0.25,
    max_connections: int | None = None,
) -> dict[str, np.ndarray]:
    """Create a sparse source-based AMPA/GABA synapse table.

    The rule is deliberately conservative and transparent: source E projects AMPA;
    source PV/SST/VIP projects GABA.  Connection probability decays with Euclidean
    distance and is modulated by a small source/target prior.
    """
    if length_constant_mm <= 0.0:
        raise ValueError("length_constant_mm must be positive")
    if plasticity_coefficient < 0.0:
        raise ValueError("plasticity_coefficient must be nonnegative")
    n = int(positions_mm.shape[0])
    if n != int(cell_type_index.shape[0]):
        raise ValueError("positions_mm and cell_type_index length mismatch")

    sources: list[int] = []
    targets: list[int] = []
    weights: list[float] = []
    receptor: list[int] = []  # 0 AMPA, 1 GABA
    plasticity: list[float] = []

    for pre in range(n):
        pre_type = cell_types[int(cell_type_index[pre])]
        diffs = positions_mm - positions_mm[pre]
        dist = np.sqrt(np.sum(diffs * diffs, axis=1))
        for post in range(n):
            if pre == post:
                continue
            post_type = cell_types[int(cell_type_index[post])]
            p0 = _default_connection_probability(pre_type, post_type)
            p = p0 * float(np.exp(-dist[post] / length_constant_mm))
            if rng.random() <= p:
                sources.append(pre)
                targets.append(post)
                is_ampa = pre_type == "E"
                receptor.append(0 if is_ampa else 1)
                # Weight units are arbitrary/native until a simulator backend calibrates them.
                weights.append(0.01 if is_ampa else 0.02)
                plasticity.append(float(plasticity_coefficient))

    if max_connections is not None and len(sources) > max_connections:
        idx = rng.choice(len(sources), size=max_connections, replace=False)
        sources = [sources[i] for i in idx]
        targets = [targets[i] for i in idx]
        weights = [weights[i] for i in idx]
        receptor = [receptor[i] for i in idx]
        plasticity = [plasticity[i] for i in idx]

    return {
        "source": np.asarray(sources, dtype=np.int32),
        "target": np.asarray(targets, dtype=np.int32),
        "weight": np.asarray(weights, dtype=float),
        "receptor_index": np.asarray(receptor, dtype=np.int8),
        "plasticity_coefficient": np.asarray(plasticity, dtype=float),
        "receptor_names": np.asarray(["AMPA", "GABA"], dtype=object),
    }


def make_cortex_network(
    XYZ_mm: Sequence[float],
    N: int,
    Ls: Sequence[float],
    Ld: Sequence[Sequence[float]],
    *,
    model_family: ModelFamily = "tfne-izhikevich",
    plasticity_coefficient: float = 0.0,
    seed: int = 0,
    min_distance_um: float = 10.0,
    max_attempts_per_neuron: int = 10_000,
    connection_length_constant_mm: float = 0.25,
    make_synapses: bool = True,
    max_connections: int | None = None,
    tfne_source_radius_um: float = 25.0,
    izh_current_to_ampere_scale: float = 1e-12,
) -> CortexNetworkSpec:
    """Create a layered 3-D cortical network specification.

    Parameters
    ----------
    XYZ_mm:
        Physical volume size ``[X, Y, Z]`` in millimetres.
    N:
        Number of neurons.
    Ls:
        Relative layer sizes along the z-axis. Values are normalized and must have
        positive sum; the common case is that they sum to 1.
    Ld:
        Per-layer cell-type density rows in ``[E, PV, SST, VIP]`` order. Rows may be
        percentages such as ``[75, 15, 9, 1]`` or fractions such as ``[0.75, 0.15, 0.09, 0.01]``.
    model_family:
        ``izhikevich`` for point-neuron metadata only, ``tfne`` for field/source metadata,
        or ``tfne-izhikevich`` for the hybrid bridge.
    plasticity_coefficient:
        Nonnegative synaptic plasticity gain. ``0`` means static synapses; values above
        ``1`` are allowed but remain user-declared and should be bounded by optimization
        manifests.
    min_distance_um:
        Minimum centre-to-centre separation used for non-overlap, in micrometres.

    Notes
    -----
    Layer neuron counts are allocated by layer volume fraction ``Ls``.  Cell-type counts
    inside each layer are allocated by ``Ld`` using largest-remainder rounding so that the
    total count is exactly ``N``.
    """
    if N <= 0:
        raise ValueError("N must be positive")
    if model_family not in {"izhikevich", "tfne", "tfne-izhikevich"}:
        raise ValueError("model_family must be 'izhikevich', 'tfne', or 'tfne-izhikevich'")
    if plasticity_coefficient < 0.0:
        raise ValueError("plasticity_coefficient must be nonnegative")
    if min_distance_um < 0.0:
        raise ValueError("min_distance_um must be nonnegative")
    if tfne_source_radius_um <= 0.0:
        raise ValueError("tfne_source_radius_um must be positive")
    if izh_current_to_ampere_scale <= 0.0:
        raise ValueError("izh_current_to_ampere_scale must be positive")

    xyz = _as_float_array("XYZ_mm", XYZ_mm, expected_len=3)
    if np.any(xyz <= 0.0):
        raise ValueError("XYZ_mm entries must be positive")
    ls = _normalize_simplex("Ls", Ls)
    ld_frac = _normalize_layer_density(Ld, n_layers=int(ls.shape[0]))

    z_edges = np.concatenate([[0.0], np.cumsum(ls) * xyz[2]])
    layer_z_bounds = tuple((float(z_edges[i]), float(z_edges[i + 1])) for i in range(len(ls)))
    counts = _allocate_counts(N, ls, ld_frac)

    rng = np.random.default_rng(seed)
    positions_mm, layer_index, cell_type_index = _sample_non_overlapping_positions(
        rng=rng,
        xyz_mm=xyz,
        layer_z_bounds_mm=layer_z_bounds,
        counts=counts,
        min_distance_mm=min_distance_um * 1e-3,
        max_attempts_per_neuron=max_attempts_per_neuron,
    )

    synapses: Mapping[str, np.ndarray]
    if make_synapses:
        synapses = make_distance_synapses(
            positions_mm,
            cell_type_index,
            rng=rng,
            plasticity_coefficient=plasticity_coefficient,
            length_constant_mm=connection_length_constant_mm,
            max_connections=max_connections,
        )
    else:
        synapses = {
            "source": np.asarray([], dtype=np.int32),
            "target": np.asarray([], dtype=np.int32),
            "weight": np.asarray([], dtype=float),
            "receptor_index": np.asarray([], dtype=np.int8),
            "plasticity_coefficient": np.asarray([], dtype=float),
            "receptor_names": np.asarray(["AMPA", "GABA"], dtype=object),
        }

    positions_m = positions_mm * 1e-3
    tfne: dict[str, Any] = {}
    if model_family in {"tfne", "tfne-izhikevich"}:
        tfne = {
            "source_positions_m": positions_m,
            "source_radii_m": np.full(N, tfne_source_radius_um * 1e-6, dtype=float),
            "izh_current_to_ampere_scale": float(izh_current_to_ampere_scale),
            "csd_sign_convention": "positive CSD enters extracellular domain",
        }

    return CortexNetworkSpec(
        model_family=model_family,
        xyz_mm=tuple(float(x) for x in xyz),
        layer_size_fraction=tuple(float(x) for x in ls),
        layer_z_bounds_mm=layer_z_bounds,
        layer_density_fraction=tuple(tuple(float(x) for x in row) for row in ld_frac),
        plasticity_coefficient=float(plasticity_coefficient),
        min_distance_mm=float(min_distance_um * 1e-3),
        seed=int(seed),
        positions_mm=positions_mm,
        positions_m=positions_m,
        layer_index=layer_index,
        cell_type_index=cell_type_index,
        cell_types=CELL_ORDER,
        synapses=synapses,
        tfne=tfne,
    )


def make_cortex_network_json(path: str | Path, network: CortexNetworkSpec) -> Path:
    """Write a generated network specification to JSON and return the path."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(network.to_json_dict(), indent=2), encoding="utf-8")
    return out
