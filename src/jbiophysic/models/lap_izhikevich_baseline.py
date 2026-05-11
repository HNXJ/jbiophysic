"""LAP-driven Izhikevich spontaneous baseline scaffold.

This module builds reduced point-neuron populations from LAP laminar marker counts
and runs bounded spontaneous Izhikevich baseline activity. It is exploratory
computational infrastructure only: no sensory input, no omission condition, no
top-down prediction, no TFNE field amplitude validation, and no biological proof.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple

import numpy as np

from jbiophysic.data.lap import (
    LAPCountRow,
    extract_lap_layer_counts,
    summarize_lap_counts,
)

try:  # Avoid making JAX a hard dependency for this NumPy baseline scaffold.
    from jbiophysic.cells.izhikevich import (  # type: ignore
        FAST_SPIKING,
        LOW_THRESHOLD_SPIKING,
        REGULAR_SPIKING,
    )
except Exception:  # pragma: no cover - used only in minimal non-JAX installs.
    class _Params(NamedTuple):
        a: float
        b: float
        c: float
        d: float
        v_spike_mV: float = 30.0

    REGULAR_SPIKING = _Params(a=0.02, b=0.20, c=-65.0, d=8.0)
    FAST_SPIKING = _Params(a=0.10, b=0.20, c=-65.0, d=2.0)
    LOW_THRESHOLD_SPIKING = _Params(a=0.02, b=0.25, c=-65.0, d=2.0)


MARKER_TO_MODEL_CELL_TYPE: dict[str, str] = {
    "NG": "E",
    "PV": "PV",
    "CB": "CB_LTS",
    "CR": "CR_LTS_or_VIP_proxy",
}

INHIBITORY_MODEL_TYPES = {"PV", "CB_LTS", "CR_LTS_or_VIP_proxy"}


@dataclass(frozen=True)
class LAPBaselineConfig:
    mat_path: Path
    seed: int = 17
    t_ms: float = 1000.0
    dt_ms: float = 0.1
    neurons_per_area: int = 100
    tube_radius_m: float = 0.1e-3
    area_spacing_m: float = 0.55e-3
    min_separation_m: float = 4.0e-6
    include_areas: tuple[str, ...] | None = None
    connectivity: str = "sparse_local"
    mean_in_degree: int = 40
    baseline_mode: str = "spontaneous"


@dataclass(frozen=True)
class LAPPopulation:
    neuron_id: np.ndarray
    area: np.ndarray
    layer: np.ndarray
    marker: np.ndarray
    cell_type: np.ndarray
    x_m: np.ndarray
    y_m: np.ndarray
    z_m: np.ndarray
    a: np.ndarray
    b: np.ndarray
    c: np.ndarray
    d: np.ndarray
    v_spike_mV: np.ndarray
    base_current: np.ndarray
    noise_sigma: np.ndarray
    raw_lap_count: np.ndarray


def _params_for_model_cell_type(cell_type: str):
    if cell_type == "E":
        return REGULAR_SPIKING
    if cell_type == "PV":
        return FAST_SPIKING
    if cell_type in {"CB_LTS", "CR_LTS_or_VIP_proxy"}:
        return LOW_THRESHOLD_SPIKING
    raise ValueError(f"unknown model cell_type: {cell_type}")


def _area_order(rows: Sequence[LAPCountRow], include_areas: tuple[str, ...] | None) -> list[str]:
    observed = sorted({row.area for row in rows})
    if include_areas is None:
        return observed
    missing = [area for area in include_areas if area not in observed]
    if missing:
        raise ValueError(f"include_areas contains unknown LAP areas: {missing}")
    return list(include_areas)


def _hamilton_allocate(raw_counts: np.ndarray, total: int) -> np.ndarray:
    if total <= 0:
        raise ValueError("neurons_per_area must be positive")
    if raw_counts.ndim != 1:
        raise ValueError("raw_counts must be one-dimensional")
    if not np.all(np.isfinite(raw_counts)) or np.any(raw_counts < 0):
        raise ValueError("raw_counts must be finite and non-negative")
    if float(np.sum(raw_counts)) <= 0.0:
        # Deterministic fallback for pathological all-zero synthetic areas.
        out = np.zeros_like(raw_counts, dtype=int)
        out[0] = total
        return out

    ideal = raw_counts / float(np.sum(raw_counts)) * int(total)
    allocation = np.floor(ideal).astype(int)
    # Preserve true zero bins as zero because their ideal allocation is exactly 0.
    remaining = int(total - int(np.sum(allocation)))
    if remaining > 0:
        remainders = ideal - allocation
        # Stable tie-break: larger remainder first, then larger raw count, then lower index.
        order = sorted(
            range(len(raw_counts)),
            key=lambda i: (float(remainders[i]), float(raw_counts[i]), -i),
            reverse=True,
        )
        for idx in order[:remaining]:
            if raw_counts[idx] > 0:
                allocation[idx] += 1
        # If all nonzero bins were exhausted by numerical oddities, finish deterministically.
        while int(np.sum(allocation)) < total:
            idx = int(np.argmax(raw_counts))
            allocation[idx] += 1
    elif remaining < 0:
        order = sorted(
            range(len(raw_counts)),
            key=lambda i: (float(ideal[i] - allocation[i]), float(raw_counts[i]), -i),
        )
        for idx in order:
            if int(np.sum(allocation)) == total:
                break
            if allocation[idx] > 0:
                allocation[idx] -= 1
    if int(np.sum(allocation)) != total:
        raise RuntimeError("Hamilton allocation failed to preserve area total")
    return allocation


def allocate_integer_counts_from_lap(
    rows: Sequence[LAPCountRow],
    neurons_per_area: int,
    include_areas: tuple[str, ...] | None = None,
) -> list[dict[str, object]]:
    """Allocate reduced integer neuron counts from LAP float layer counts.

    Allocation is per area, using largest-remainder/Hamilton apportionment, so
    each included area has exactly `neurons_per_area` allocated neurons.
    """
    if neurons_per_area <= 0:
        raise ValueError("neurons_per_area must be positive")
    areas = _area_order(rows, include_areas)
    out: list[dict[str, object]] = []
    for area in areas:
        area_rows = [row for row in rows if row.area == area]
        if not area_rows:
            raise ValueError(f"no LAP rows found for area {area}")
        raw = np.asarray([max(float(row.count), 0.0) for row in area_rows], dtype=float)
        allocated = _hamilton_allocate(raw, neurons_per_area)
        for row, count in zip(area_rows, allocated, strict=True):
            model_cell_type = MARKER_TO_MODEL_CELL_TYPE.get(row.marker, row.marker)
            out.append(
                {
                    "area": row.area,
                    "marker": row.marker,
                    "layer_index": int(row.layer_index),
                    "layer_label": row.layer_label,
                    "raw_count": float(row.count),
                    "allocated_count": int(count),
                    "model_cell_type": model_cell_type,
                }
            )
    return out


def _sample_layer_positions(
    rng: np.random.Generator,
    *,
    n: int,
    center_x_m: float,
    center_y_m: float,
    radius_m: float,
    z0_m: float,
    z1_m: float,
    existing: list[np.ndarray],
    min_separation_m: float,
    max_attempts: int = 100_000,
) -> np.ndarray:
    points: list[np.ndarray] = []
    attempts = 0
    while len(points) < n and attempts < max_attempts:
        attempts += 1
        r = radius_m * math.sqrt(float(rng.uniform()))
        theta = float(rng.uniform(0.0, 2.0 * math.pi))
        candidate = np.asarray(
            [
                center_x_m + r * math.cos(theta),
                center_y_m + r * math.sin(theta),
                float(rng.uniform(z0_m, z1_m)),
            ],
            dtype=float,
        )
        if existing or points:
            all_existing = np.asarray(existing + points, dtype=float)
            if float(np.min(np.linalg.norm(all_existing - candidate, axis=1))) < min_separation_m:
                continue
        points.append(candidate)
    if len(points) != n:
        raise RuntimeError(
            f"Could not sample {n} non-overlapping LAP positions; "
            "reduce min_separation_m or neurons_per_area."
        )
    return np.asarray(points, dtype=float)


def _layer_bounds_by_area(
    allocations: Sequence[Mapping[str, object]],
) -> dict[str, dict[int, tuple[float, float]]]:
    bounds: dict[str, dict[int, tuple[float, float]]] = {}
    for area in sorted({str(row["area"]) for row in allocations}):
        layer_indices = sorted(
            {int(row["layer_index"]) for row in allocations if str(row["area"]) == area}
        )
        n_layers = len(layer_indices)
        area_bounds: dict[int, tuple[float, float]] = {}
        for rank, layer_index in enumerate(layer_indices):
            area_bounds[layer_index] = (rank / n_layers, (rank + 1) / n_layers)
        bounds[area] = area_bounds
    return bounds


def build_lap_population(cfg: LAPBaselineConfig) -> LAPPopulation:
    """Build a reduced LAP-based Izhikevich population with area/layer metadata."""
    if cfg.baseline_mode != "spontaneous":
        raise ValueError("Only spontaneous baseline mode is supported in this scaffold")
    rows = extract_lap_layer_counts(cfg.mat_path)
    allocations = allocate_integer_counts_from_lap(rows, cfg.neurons_per_area, cfg.include_areas)
    areas = sorted({str(row["area"]) for row in allocations})
    layer_bounds = _layer_bounds_by_area(allocations)
    rng = np.random.default_rng(cfg.seed)

    neuron_id: list[int] = []
    area_values: list[str] = []
    layer_values: list[str] = []
    marker_values: list[str] = []
    cell_types: list[str] = []
    raw_counts: list[float] = []
    positions: list[np.ndarray] = []
    existing: list[np.ndarray] = []

    next_id = 0
    area_to_x = {
        area: (i - (len(areas) - 1) / 2.0) * cfg.area_spacing_m for i, area in enumerate(areas)
    }

    for area in areas:
        for row in [r for r in allocations if str(r["area"]) == area]:
            count = int(row["allocated_count"])
            if count <= 0:
                continue
            layer_index = int(row["layer_index"])
            z0_frac, z1_frac = layer_bounds[area][layer_index]
            pts = _sample_layer_positions(
                rng,
                n=count,
                center_x_m=area_to_x[area],
                center_y_m=0.0,
                radius_m=cfg.tube_radius_m,
                z0_m=z0_frac * 1.0e-3,
                z1_m=z1_frac * 1.0e-3,
                existing=existing,
                min_separation_m=cfg.min_separation_m,
            )
            existing.extend([p for p in pts])
            positions.append(pts)
            for _ in range(count):
                neuron_id.append(next_id)
                area_values.append(area)
                layer_values.append(str(row["layer_label"]))
                marker_values.append(str(row["marker"]))
                cell_types.append(str(row["model_cell_type"]))
                raw_counts.append(float(row["raw_count"]))
                next_id += 1

    if not positions:
        raise ValueError("LAP allocation produced an empty population")
    xyz = np.vstack(positions)
    cell_type_arr = np.asarray(cell_types, dtype="U32")
    marker_arr = np.asarray(marker_values, dtype="U8")
    params = [_params_for_model_cell_type(str(cell_type)) for cell_type in cell_type_arr]

    a = np.asarray([float(p.a) for p in params], dtype=np.float64)
    b = np.asarray([float(p.b) for p in params], dtype=np.float64)
    c = np.asarray([float(p.c) for p in params], dtype=np.float64)
    d = np.asarray([float(p.d) for p in params], dtype=np.float64)
    v_spike = np.asarray([float(getattr(p, "v_spike_mV", 30.0)) for p in params], dtype=np.float64)

    base = np.empty(len(cell_type_arr), dtype=np.float64)
    sigma = np.empty(len(cell_type_arr), dtype=np.float64)
    is_e = cell_type_arr == "E"
    is_pv = cell_type_arr == "PV"
    is_lts = ~(is_e | is_pv)
    base[is_e] = rng.uniform(5.5, 7.5, size=int(np.sum(is_e)))
    base[is_pv] = rng.uniform(5.0, 7.0, size=int(np.sum(is_pv)))
    base[is_lts] = rng.uniform(4.8, 6.8, size=int(np.sum(is_lts)))
    sigma[is_e] = 1.1
    sigma[is_pv] = 1.0
    sigma[is_lts] = 0.9

    return LAPPopulation(
        neuron_id=np.asarray(neuron_id, dtype=np.int64),
        area=np.asarray(area_values, dtype="U16"),
        layer=np.asarray(layer_values, dtype="U8"),
        marker=marker_arr,
        cell_type=cell_type_arr,
        x_m=xyz[:, 0].astype(np.float64),
        y_m=xyz[:, 1].astype(np.float64),
        z_m=xyz[:, 2].astype(np.float64),
        a=a,
        b=b,
        c=c,
        d=d,
        v_spike_mV=v_spike,
        base_current=base,
        noise_sigma=sigma,
        raw_lap_count=np.asarray(raw_counts, dtype=np.float64),
    )


def build_sparse_baseline_weights(pop: LAPPopulation, cfg: LAPBaselineConfig) -> np.ndarray:
    """Build sparse post x pre native-current synaptic weights with no autapses."""
    if cfg.connectivity != "sparse_local":
        raise ValueError("Only sparse_local connectivity is currently implemented")
    rng = np.random.default_rng(cfg.seed + 10_000)
    n = len(pop.neuron_id)
    w = np.zeros((n, n), dtype=np.float32)
    if n <= 1:
        return w
    in_degree = max(0, min(int(cfg.mean_in_degree), n - 1))
    if in_degree == 0:
        return w

    positions = np.column_stack([pop.x_m, pop.y_m, pop.z_m])
    for post in range(n):
        distances = np.linalg.norm(positions - positions[post], axis=1)
        distances[post] = np.inf
        # Prefer local candidates but retain some inter-area possibility when many neurons exist.
        nearest = np.argsort(distances)[: max(in_degree * 3, in_degree)]
        pre_ids = rng.choice(nearest, size=in_degree, replace=False)
        for pre in pre_ids:
            ctype = str(pop.cell_type[pre])
            if ctype == "E":
                w[post, pre] = rng.uniform(0.0, 0.08)
            elif ctype == "PV":
                w[post, pre] = rng.uniform(-0.35, 0.0)
            else:
                w[post, pre] = rng.uniform(-0.25, 0.0)
    np.fill_diagonal(w, 0.0)
    return w


def run_lap_spontaneous_baseline(
    pop: LAPPopulation,
    weights: np.ndarray,
    cfg: LAPBaselineConfig,
) -> dict[str, np.ndarray]:
    """Run bounded spontaneous Izhikevich baseline activity.

    There is no sensory, omission, task, or top-down prediction input in this run.
    """
    if cfg.t_ms <= 0 or cfg.dt_ms <= 0:
        raise ValueError("t_ms and dt_ms must be positive")
    n = len(pop.neuron_id)
    weights = np.asarray(weights, dtype=np.float64)
    if weights.shape != (n, n):
        raise ValueError(f"weights must have shape {(n, n)}, got {weights.shape}")
    if np.any(np.diag(weights) != 0):
        raise ValueError("weights must not contain autapses")

    rng = np.random.default_rng(cfg.seed + 20_000)
    n_steps = int(round(cfg.t_ms / cfg.dt_ms))
    v = rng.uniform(-70.0, -60.0, size=n)
    u = pop.b * v
    syn_state = np.zeros(n, dtype=np.float64)
    tau_pre_ms = np.where(pop.cell_type == "E", 5.0, np.where(pop.cell_type == "PV", 8.0, 25.0))
    syn_decay = np.exp(-cfg.dt_ms / tau_pre_ms)

    spikes = np.zeros((n_steps, n), dtype=np.bool_)
    voltage_mV = np.zeros((n_steps, n), dtype=np.float32)

    for k in range(n_steps):
        noise = rng.normal(0.0, pop.noise_sigma, size=n)
        current_in = pop.base_current + noise + weights @ syn_state
        dv = 0.04 * v * v + 5.0 * v + 140.0 - u + current_in
        du = pop.a * (pop.b * v - u)
        v = v + cfg.dt_ms * dv
        u = u + cfg.dt_ms * du
        spiked = v >= pop.v_spike_mV
        spikes[k, :] = spiked
        voltage_mV[k, :] = np.where(spiked, pop.v_spike_mV, v).astype(np.float32)
        v = np.where(spiked, pop.c, v)
        u = np.where(spiked, u + pop.d, u)
        syn_state = syn_state * syn_decay + spiked.astype(np.float64)

    if not np.all(np.isfinite(voltage_mV)):
        raise FloatingPointError("voltage_mV contains NaN/Inf")
    return {
        "spikes": spikes,
        "voltage_mV": voltage_mV,
        "sensory_input_enabled": np.asarray(False),
        "omission_input_enabled": np.asarray(False),
        "top_down_prediction_enabled": np.asarray(False),
    }


def _nested_counts(pop: LAPPopulation) -> dict[str, dict[str, dict[str, int]]]:
    out: dict[str, dict[str, dict[str, int]]] = {}
    for area in sorted(set(map(str, pop.area))):
        out[area] = {}
        area_mask = pop.area == area
        for layer in sorted(set(map(str, pop.layer[area_mask]))):
            layer_mask = area_mask & (pop.layer == layer)
            out[area][layer] = {}
            for marker in sorted(set(map(str, pop.marker[layer_mask]))):
                out[area][layer][marker] = int(np.sum(layer_mask & (pop.marker == marker)))
    return out


def summarize_lap_baseline(
    pop: LAPPopulation,
    result: Mapping[str, np.ndarray],
    cfg: LAPBaselineConfig,
) -> dict[str, object]:
    spikes = np.asarray(result["spikes"])
    spike_counts = spikes.sum(axis=0).astype(int)
    mat_summary = summarize_lap_counts(cfg.mat_path)
    warnings = list(mat_summary.get("warnings", []))

    rates_by_marker: dict[str, float] = {}
    duration_s = cfg.t_ms / 1000.0
    for marker in sorted(set(map(str, pop.marker))):
        mask = pop.marker == marker
        n_marker = np.sum(mask)
        rates_by_marker[marker] = float(
            np.sum(spike_counts[mask]) / max(n_marker * duration_s, 1e-12)
        )

    return {
        "truth_status": "truth_safe_unverified",
        "baseline_mode": cfg.baseline_mode,
        "mat_file": Path(cfg.mat_path).name,
        "seed": int(cfg.seed),
        "t_ms": float(cfg.t_ms),
        "dt_ms": float(cfg.dt_ms),
        "n_areas": int(len(set(map(str, pop.area)))) ,
        "neurons_per_area": int(cfg.neurons_per_area),
        "total_neurons": int(len(pop.neuron_id)),
        "counts_by_area": {
            area: int(np.sum(pop.area == area)) for area in sorted(set(map(str, pop.area)))
        },
        "counts_by_marker": {
            marker: int(np.sum(pop.marker == marker))
            for marker in sorted(set(map(str, pop.marker)))
        },
        "counts_by_area_layer_marker": _nested_counts(pop),
        "spike_floor": {
            "silent_neuron_count": int(np.sum(spike_counts == 0)),
            "min_spikes": int(np.min(spike_counts)) if len(spike_counts) else 0,
            "median_spikes": float(np.median(spike_counts)) if len(spike_counts) else 0.0,
            "max_spikes": int(np.max(spike_counts)) if len(spike_counts) else 0,
        },
        "rates_hz_by_marker": rates_by_marker,
        "warnings": sorted(set(warnings)),
        "units": {
            "time": "ms",
            "voltage": "mV",
            "position": "m",
            "izhikevich_current": "native/current-like units, not nA",
        },
        "claim_status": {
            "level": "computational spontaneous baseline scaffold",
            "biological_validation": False,
            "lfp_csd_amplitude_validation": False,
        },
        "inputs_enabled": {
            "sensory": bool(np.asarray(result.get("sensory_input_enabled", False)).item()),
            "omission": bool(np.asarray(result.get("omission_input_enabled", False)).item()),
            "top_down_prediction": bool(
                np.asarray(result.get("top_down_prediction_enabled", False)).item()
            ),
        },
        "marker_to_model_mapping": dict(MARKER_TO_MODEL_CELL_TYPE),
    }


def _write_neurons_csv(path: Path, pop: LAPPopulation) -> None:
    fields = [
        "neuron_id",
        "area",
        "layer",
        "marker",
        "cell_type",
        "x_m",
        "y_m",
        "z_m",
        "a",
        "b",
        "c",
        "d",
        "v_spike_mV",
        "base_current_native",
        "noise_sigma_native",
        "raw_lap_count",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for i in range(len(pop.neuron_id)):
            writer.writerow(
                {
                    "neuron_id": int(pop.neuron_id[i]),
                    "area": str(pop.area[i]),
                    "layer": str(pop.layer[i]),
                    "marker": str(pop.marker[i]),
                    "cell_type": str(pop.cell_type[i]),
                    "x_m": float(pop.x_m[i]),
                    "y_m": float(pop.y_m[i]),
                    "z_m": float(pop.z_m[i]),
                    "a": float(pop.a[i]),
                    "b": float(pop.b[i]),
                    "c": float(pop.c[i]),
                    "d": float(pop.d[i]),
                    "v_spike_mV": float(pop.v_spike_mV[i]),
                    "base_current_native": float(pop.base_current[i]),
                    "noise_sigma_native": float(pop.noise_sigma[i]),
                    "raw_lap_count": float(pop.raw_lap_count[i]),
                }
            )


def _write_rates_csv(
    path: Path, pop: LAPPopulation, spikes: np.ndarray, cfg: LAPBaselineConfig
) -> None:
    duration_s = cfg.t_ms / 1000.0
    fields = ["area", "layer", "marker", "cell_type", "n_neurons", "spike_count", "rate_hz"]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        keys = sorted(
            set(
                zip(
                    map(str, pop.area),
                    map(str, pop.layer),
                    map(str, pop.marker),
                    map(str, pop.cell_type),
                    strict=False,
                )
            )
        )
        for area, layer, marker, cell_type in keys:
            mask = (
                (pop.area == area)
                & (pop.layer == layer)
                & (pop.marker == marker)
                & (pop.cell_type == cell_type)
            )
            n = int(np.sum(mask))
            spike_count = int(np.sum(spikes[:, mask]))
            writer.writerow(
                {
                    "area": area,
                    "layer": layer,
                    "marker": marker,
                    "cell_type": cell_type,
                    "n_neurons": n,
                    "spike_count": spike_count,
                    "rate_hz": float(spike_count / max(n * duration_s, 1e-12)),
                }
            )


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_lap_baseline_outputs(
    out_dir: str | Path,
    pop: LAPPopulation,
    weights: np.ndarray,
    result: Mapping[str, np.ndarray],
    cfg: LAPBaselineConfig,
) -> dict[str, str]:
    """Write baseline arrays, tables, summary, and SHA256 manifest."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    _write_neurons_csv(out / "neurons.csv", pop)
    np.savez_compressed(
        out / "spikes_and_voltage.npz",
        spikes=result["spikes"],
        voltage_mV=result["voltage_mV"],
    )
    np.savez_compressed(out / "weights_post_pre.npz", weights_post_pre=np.asarray(weights))
    _write_rates_csv(
        out / "population_rates_by_area_layer_marker.csv",
        pop,
        np.asarray(result["spikes"]),
        cfg,
    )
    summary = summarize_lap_baseline(pop, result, cfg)
    with (out / "summary.json").open("w") as f:
        json.dump(summary, f, indent=2)

    files = [
        out / "neurons.csv",
        out / "spikes_and_voltage.npz",
        out / "weights_post_pre.npz",
        out / "population_rates_by_area_layer_marker.csv",
        out / "summary.json",
    ]
    with (out / "hashes.sha256").open("w") as f:
        for path in files:
            f.write(f"{_sha256(path)}  {path.name}\n")
    files.append(out / "hashes.sha256")
    return {path.name: str(path) for path in files}


__all__ = [
    "LAPBaselineConfig",
    "LAPPopulation",
    "MARKER_TO_MODEL_CELL_TYPE",
    "allocate_integer_counts_from_lap",
    "build_lap_population",
    "build_sparse_baseline_weights",
    "run_lap_spontaneous_baseline",
    "summarize_lap_baseline",
    "write_lap_baseline_outputs",
]
