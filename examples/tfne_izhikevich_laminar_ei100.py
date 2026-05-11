#!/usr/bin/env python3
"""
TFNE-Izhikevich 100-neuron laminar E/PV/SST smoke simulation.

Purpose
-------
Start-point simulation for a three-layer cortical tube:
  - 100 point neurons: 75 E, 15 PV, 10 SST
  - three depth compartments: superficial, mid, deep
  - Izhikevich dynamics with noisy all-to-all random synapses
  - TFNE-style source projection into a 3-D cylindrical extracellular volume
  - gauge-fixed smoke Poisson field snapshots

Scientific status
-----------------
Exploratory scaffold only. This is not a validated biological simulator and does
not claim biological truth. Izhikevich current is native/current-like drive; only
explicit calibration constants map spike-derived source amplitudes to SI amperes
for TFNE projection.

Run from the jbiophysic repo root or standalone:
    PYTHONPATH=src python examples/tfne_izhikevich_laminar_ei100.py \
        --out outputs/tfne_izhikevich_laminar_ei100
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import NamedTuple

import numpy as np

try:  # Prefer current jbiophysic constants when run inside the repo.
    from jbiophysic.cells.izhikevich import (  # type: ignore
        FAST_SPIKING,
        LOW_THRESHOLD_SPIKING,
        REGULAR_SPIKING,
    )

    HAVE_JBIOPHYSIC = True
except Exception:  # Standalone fallback, identical canonical parameter values.
    HAVE_JBIOPHYSIC = False

    class _Params(NamedTuple):
        a: float
        b: float
        c: float
        d: float
        v_spike_mV: float = 30.0

    REGULAR_SPIKING = _Params(a=0.02, b=0.20, c=-65.0, d=8.0)
    FAST_SPIKING = _Params(a=0.10, b=0.20, c=-65.0, d=2.0)
    LOW_THRESHOLD_SPIKING = _Params(a=0.02, b=0.25, c=-65.0, d=2.0)


@dataclass(frozen=True)
class SimConfig:
    seed: int = 17
    t_ms: float = 1000.0
    dt_ms: float = 0.1
    tube_depth_m: float = 1.0e-3
    tube_radius_m: float = 0.1e-3
    layer_depths_m: tuple[float, float, float] = (0.3e-3, 0.1e-3, 0.6e-3)
    grid_h_m: float = 25.0e-6
    source_radius_m: float = 20.0e-6
    conductivity_s_m: float = 0.30
    poisson_steps: int = 120
    field_stride_ms: float = 10.0
    rate_bin_ms: float = 1.0
    calibration_attempts: int = 8
    silent_bias_increment: float = 1.5
    syn_seed_offset: int = 10_000


@dataclass(frozen=True)
class LayerSpec:
    name: str
    z0_m: float
    z1_m: float
    n_e: int
    n_pv: int
    n_sst: int


@dataclass(frozen=True)
class Population:
    neuron_id: np.ndarray
    layer: np.ndarray
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


@dataclass(frozen=True)
class TFNEGrid:
    x_m: np.ndarray
    y_m: np.ndarray
    z_m: np.ndarray
    coords_m: np.ndarray
    active_mask: np.ndarray
    voxel_volume_m3: float
    h_m: float


def build_layer_specs(cfg: SimConfig) -> list[LayerSpec]:
    d_sup, d_mid, d_deep = cfg.layer_depths_m
    if not math.isclose(d_sup + d_mid + d_deep, cfg.tube_depth_m, rel_tol=0.0, abs_tol=1e-15):
        raise ValueError("Layer depths must sum to total tube depth.")
    return [
        LayerSpec("superficial", 0.0, d_sup, n_e=30, n_pv=4, n_sst=7),
        LayerSpec("mid", d_sup, d_sup + d_mid, n_e=5, n_pv=5, n_sst=0),
        LayerSpec("deep", d_sup + d_mid, cfg.tube_depth_m, n_e=40, n_pv=6, n_sst=3),
    ]


def _params_for_type(cell_type: str):
    if cell_type == "E":
        return REGULAR_SPIKING
    if cell_type == "PV":
        return FAST_SPIKING
    if cell_type == "SST":
        return LOW_THRESHOLD_SPIKING
    raise ValueError(f"unknown cell type: {cell_type}")


def _uniform_cylinder_positions(
    rng: np.random.Generator,
    n: int,
    radius_m: float,
    z0_m: float,
    z1_m: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    # sqrt sampling gives uniform density over the disk area.
    r = radius_m * np.sqrt(rng.uniform(0.0, 1.0, size=n))
    theta = rng.uniform(0.0, 2.0 * np.pi, size=n)
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    z = rng.uniform(z0_m, z1_m, size=n)
    return x, y, z


def build_population(cfg: SimConfig) -> Population:
    rng = np.random.default_rng(cfg.seed)
    neuron_id: list[int] = []
    layer: list[str] = []
    cell_type: list[str] = []
    xs: list[float] = []
    ys: list[float] = []
    zs: list[float] = []

    next_id = 0
    # Ordered by layer and type, preserving the user-specified counts exactly.
    for spec in build_layer_specs(cfg):
        for t_name, count in (("E", spec.n_e), ("PV", spec.n_pv), ("SST", spec.n_sst)):
            if count == 0:
                continue
            x, y, z = _uniform_cylinder_positions(
                rng, count, cfg.tube_radius_m, spec.z0_m, spec.z1_m
            )
            for i in range(count):
                neuron_id.append(next_id)
                layer.append(spec.name)
                cell_type.append(t_name)
                xs.append(float(x[i]))
                ys.append(float(y[i]))
                zs.append(float(z[i]))
                next_id += 1

    cell_type_arr = np.asarray(cell_type, dtype="U3")
    layer_arr = np.asarray(layer, dtype="U16")
    params = [_params_for_type(str(t)) for t in cell_type_arr]

    a = np.asarray([p.a for p in params], dtype=np.float64)
    b = np.asarray([p.b for p in params], dtype=np.float64)
    c = np.asarray([p.c for p in params], dtype=np.float64)
    d = np.asarray([p.d for p in params], dtype=np.float64)
    v_spike = np.asarray([getattr(p, "v_spike_mV", 30.0) for p in params], dtype=np.float64)

    # Current-like native Izhikevich drive. Superficial receives a small offset to
    # support denser/higher-frequency superficial dynamics; deep receives slightly less.
    base = np.empty(len(cell_type_arr), dtype=np.float64)
    sigma = np.empty(len(cell_type_arr), dtype=np.float64)
    base[cell_type_arr == "E"] = rng.uniform(6.0, 8.0, size=int(np.sum(cell_type_arr == "E")))
    base[cell_type_arr == "PV"] = rng.uniform(5.5, 7.5, size=int(np.sum(cell_type_arr == "PV")))
    base[cell_type_arr == "SST"] = rng.uniform(5.0, 7.0, size=int(np.sum(cell_type_arr == "SST")))
    base += np.where(layer_arr == "superficial", 1.0, np.where(layer_arr == "deep", -0.3, 0.0))

    sigma[cell_type_arr == "E"] = 1.2
    sigma[cell_type_arr == "PV"] = 1.0
    sigma[cell_type_arr == "SST"] = 0.9

    n = len(cell_type_arr)
    if n != 100:
        raise AssertionError(f"expected 100 neurons, got {n}")

    return Population(
        neuron_id=np.asarray(neuron_id, dtype=np.int64),
        layer=layer_arr,
        cell_type=cell_type_arr,
        x_m=np.asarray(xs, dtype=np.float64),
        y_m=np.asarray(ys, dtype=np.float64),
        z_m=np.asarray(zs, dtype=np.float64),
        a=a,
        b=b,
        c=c,
        d=d,
        v_spike_mV=v_spike,
        base_current=base,
        noise_sigma=sigma,
    )


def build_all_to_all_uniform_weights(pop: Population, cfg: SimConfig) -> np.ndarray:
    rng = np.random.default_rng(cfg.seed + cfg.syn_seed_offset)
    n = len(pop.neuron_id)
    w = np.zeros((n, n), dtype=np.float64)  # post x pre
    for pre in range(n):
        t = str(pop.cell_type[pre])
        if t == "E":
            w[:, pre] = rng.uniform(0.0, 0.10, size=n)
        elif t == "PV":
            w[:, pre] = rng.uniform(-0.50, 0.0, size=n)
        elif t == "SST":
            w[:, pre] = rng.uniform(-0.35, 0.0, size=n)
        else:
            raise ValueError(t)
    np.fill_diagonal(w, 0.0)  # no autapses in this starter scaffold.
    return w


def run_izhikevich_network(
    pop: Population,
    weights_post_pre: np.ndarray,
    cfg: SimConfig,
    *,
    bias_extra: np.ndarray | None = None,
    noise_seed_offset: int = 0,
) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(cfg.seed + 2 * cfg.syn_seed_offset + noise_seed_offset)
    n = len(pop.neuron_id)
    n_steps = int(round(cfg.t_ms / cfg.dt_ms))

    v = rng.uniform(-70.0, -60.0, size=n)
    u = pop.b * v
    syn_state = np.zeros(n, dtype=np.float64)

    tau_pre_ms = np.where(pop.cell_type == "E", 5.0, np.where(pop.cell_type == "PV", 8.0, 25.0))
    syn_decay = np.exp(-cfg.dt_ms / tau_pre_ms)
    if bias_extra is None:
        bias_extra = np.zeros(n, dtype=np.float64)

    spikes = np.zeros((n_steps, n), dtype=np.bool_)
    voltage_mV = np.zeros((n_steps, n), dtype=np.float32)

    base = pop.base_current + bias_extra

    for k in range(n_steps):
        noise = rng.normal(0.0, pop.noise_sigma, size=n)
        syn_current = weights_post_pre @ syn_state
        current_in = base + noise + syn_current

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

    return {"spikes": spikes, "voltage_mV": voltage_mV, "bias_extra": bias_extra}


def calibrate_until_all_fire(
    pop: Population, weights: np.ndarray, cfg: SimConfig
) -> dict[str, np.ndarray]:
    bias_extra = np.zeros(len(pop.neuron_id), dtype=np.float64)
    last_result: dict[str, np.ndarray] | None = None
    last_silent: np.ndarray | None = None

    for attempt in range(cfg.calibration_attempts):
        result = run_izhikevich_network(
            pop, weights, cfg, bias_extra=bias_extra, noise_seed_offset=attempt
        )
        counts = result["spikes"].sum(axis=0)
        silent = counts == 0
        result["calibration_attempt"] = np.asarray(attempt + 1, dtype=np.int64)
        result["silent_count"] = np.asarray(int(np.sum(silent)), dtype=np.int64)
        last_result = result
        last_silent = silent
        if not np.any(silent):
            return result
        # Increase only silent neurons for the next full rerun; this is a current
        # calibration, not a post-hoc spike insertion.
        bias_extra = bias_extra.copy()
        bias_extra[silent] += cfg.silent_bias_increment

    assert last_result is not None and last_silent is not None
    silent_ids = pop.neuron_id[last_silent].tolist()
    raise RuntimeError(
        "Spike-floor calibration failed: not every neuron fired at least once. "
        f"Silent neuron IDs after {cfg.calibration_attempts} attempts: {silent_ids}. "
        "Increase --calibration-attempts, --silent-bias-increment, or baseline drive."
    )


def make_cylindrical_tfne_grid(cfg: SimConfig) -> TFNEGrid:
    h = cfg.grid_h_m
    # Force exact inclusion of -radius, 0, +radius and z=0/depth within rounding tolerance.
    nx = int(round((2.0 * cfg.tube_radius_m) / h)) + 1
    nz = int(round(cfg.tube_depth_m / h)) + 1
    x = np.linspace(-cfg.tube_radius_m, cfg.tube_radius_m, nx, dtype=np.float64)
    y = np.linspace(-cfg.tube_radius_m, cfg.tube_radius_m, nx, dtype=np.float64)
    z = np.linspace(0.0, cfg.tube_depth_m, nz, dtype=np.float64)
    X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
    active = (cfg.tube_radius_m * cfg.tube_radius_m + 1e-30) >= (X * X + Y * Y)
    coords = np.stack([X, Y, Z], axis=-1)
    return TFNEGrid(
        x_m=x,
        y_m=y,
        z_m=z,
        coords_m=coords,
        active_mask=active,
        voxel_volume_m3=h**3,
        h_m=h,
    )


def gaussian_mollifier(grid: TFNEGrid, pos_m: np.ndarray, radius_m: float) -> np.ndarray:
    d2 = np.sum((grid.coords_m - pos_m.reshape(1, 1, 1, 3)) ** 2, axis=-1)
    raw = np.exp(-0.5 * d2 / (radius_m * radius_m))
    raw = np.where(grid.active_mask, raw, 0.0)
    denom = float(np.sum(raw * grid.voxel_volume_m3))
    if denom <= 0.0 or not np.isfinite(denom):
        raise FloatingPointError("invalid Gaussian source normalization")
    return raw / denom  # 1 / m^3


def precompute_source_kernels(pop: Population, grid: TFNEGrid, cfg: SimConfig) -> np.ndarray:
    kernels = np.zeros((len(pop.neuron_id), *grid.active_mask.shape), dtype=np.float32)
    for i in range(len(pop.neuron_id)):
        pos = np.asarray([pop.x_m[i], pop.y_m[i], pop.z_m[i]], dtype=np.float64)
        kernels[i] = gaussian_mollifier(grid, pos, cfg.source_radius_m).astype(np.float32)
    return kernels


def jacobi_poisson_neumann_smoke(
    source_A_m3: np.ndarray,
    grid: TFNEGrid,
    *,
    conductivity_s_m: float,
    steps: int,
) -> np.ndarray:
    if conductivity_s_m <= 0:
        raise ValueError("conductivity_s_m must be positive")
    if steps < 1:
        raise ValueError("steps must be positive")

    source = np.where(grid.active_mask, source_A_m3, 0.0).astype(np.float64)
    active_mean = float(np.sum(source) / max(int(np.sum(grid.active_mask)), 1))
    rhs = np.where(grid.active_mask, source - active_mean, 0.0)
    phi = np.zeros_like(rhs, dtype=np.float64)
    h2_over_sigma = (grid.h_m * grid.h_m) / conductivity_s_m

    for _ in range(steps):
        p = np.pad(phi, ((1, 1), (1, 1), (1, 1)), mode="edge")
        neighbor_sum = (
            p[2:, 1:-1, 1:-1]
            + p[:-2, 1:-1, 1:-1]
            + p[1:-1, 2:, 1:-1]
            + p[1:-1, :-2, 1:-1]
            + p[1:-1, 1:-1, 2:]
            + p[1:-1, 1:-1, :-2]
        )
        phi = (neighbor_sum + h2_over_sigma * rhs) / 6.0
        phi = np.where(grid.active_mask, phi, 0.0)
        phi_mean = float(np.sum(phi) / max(int(np.sum(grid.active_mask)), 1))
        phi = np.where(grid.active_mask, phi - phi_mean, 0.0)
    return phi.astype(np.float32)


def compute_tfne_field_snapshots(
    pop: Population,
    spikes: np.ndarray,
    grid: TFNEGrid,
    kernels: np.ndarray,
    cfg: SimConfig,
) -> dict[str, np.ndarray]:
    n_steps, n = spikes.shape
    stride_steps = max(1, int(round(cfg.field_stride_ms / cfg.dt_ms)))
    snap_steps = np.arange(0, n_steps, stride_steps, dtype=np.int64)

    # Explicit calibration from spike events to SI source amplitudes.
    # Signs encode E-dominant source vs inhibitory sink convention for this starter.
    source_amp_A = np.where(
        pop.cell_type == "E",
        30.0e-12,
        np.where(pop.cell_type == "PV", -45.0e-12, -25.0e-12),
    )
    source_tau_ms = np.where(pop.cell_type == "E", 3.0, np.where(pop.cell_type == "PV", 5.0, 20.0))
    source_decay = np.exp(-cfg.dt_ms / source_tau_ms)
    source_state = np.zeros(n, dtype=np.float64)

    q_snaps = np.zeros((len(snap_steps), *grid.active_mask.shape), dtype=np.float32)
    phi_snaps = np.zeros_like(q_snaps)
    source_current_A_snaps = np.zeros((len(snap_steps), n), dtype=np.float32)

    snap_i = 0
    for k in range(n_steps):
        source_state = source_state * source_decay + spikes[k].astype(np.float64)
        if snap_i < len(snap_steps) and k == snap_steps[snap_i]:
            currents_A = source_amp_A * source_state
            # Sum sparse neuron currents into conserved volumetric source q(x,y,z).
            q = np.tensordot(currents_A.astype(np.float32), kernels, axes=(0, 0)).astype(np.float32)
            q = np.where(grid.active_mask, q, 0.0).astype(np.float32)
            phi = jacobi_poisson_neumann_smoke(
                q,
                grid,
                conductivity_s_m=cfg.conductivity_s_m,
                steps=cfg.poisson_steps,
            )
            q_snaps[snap_i] = q
            phi_snaps[snap_i] = phi
            source_current_A_snaps[snap_i] = currents_A.astype(np.float32)
            snap_i += 1

    return {
        "snapshot_steps": snap_steps,
        "snapshot_times_ms": snap_steps.astype(np.float64) * cfg.dt_ms,
        "q_A_per_m3": q_snaps,
        "phi_V": phi_snaps,
        "source_current_A": source_current_A_snaps,
    }


def population_rate_table(
    pop: Population, spikes: np.ndarray, cfg: SimConfig
) -> list[dict[str, float | int | str]]:
    bin_steps = max(1, int(round(cfg.rate_bin_ms / cfg.dt_ms)))
    n_bins = spikes.shape[0] // bin_steps
    rows: list[dict[str, float | int | str]] = []
    for b_i in range(n_bins):
        i0 = b_i * bin_steps
        i1 = i0 + bin_steps
        t0 = i0 * cfg.dt_ms
        t1 = i1 * cfg.dt_ms
        window_s = (t1 - t0) / 1000.0
        for layer_name in ("superficial", "mid", "deep"):
            for cell_name in ("E", "PV", "SST"):
                mask = (pop.layer == layer_name) & (pop.cell_type == cell_name)
                n = int(np.sum(mask))
                if n == 0:
                    continue
                spike_count = int(np.sum(spikes[i0:i1, :][:, mask]))
                rate_hz = spike_count / (n * window_s)
                rows.append(
                    {
                        "bin_index": b_i,
                        "t0_ms": t0,
                        "t1_ms": t1,
                        "layer": layer_name,
                        "cell_type": cell_name,
                        "n_neurons": n,
                        "spike_count": spike_count,
                        "rate_hz": rate_hz,
                    }
                )
    return rows


def write_population_csv(path: Path, pop: Population, spike_counts: np.ndarray) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "neuron_id",
                "layer",
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
                "spike_count",
            ],
        )
        writer.writeheader()
        for i in range(len(pop.neuron_id)):
            writer.writerow(
                {
                    "neuron_id": int(pop.neuron_id[i]),
                    "layer": str(pop.layer[i]),
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
                    "spike_count": int(spike_counts[i]),
                }
            )


def write_rate_csv(path: Path, rows: list[dict[str, float | int | str]]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def summarize(
    pop: Population, spikes: np.ndarray, result: dict[str, np.ndarray], cfg: SimConfig
) -> dict:
    spike_counts = spikes.sum(axis=0).astype(np.int64)
    layer_summary: dict[str, dict[str, dict[str, float | int]]] = {}
    for layer_name in ("superficial", "mid", "deep"):
        layer_summary[layer_name] = {}
        for cell_name in ("E", "PV", "SST"):
            mask = (pop.layer == layer_name) & (pop.cell_type == cell_name)
            n = int(np.sum(mask))
            if n == 0:
                continue
            layer_summary[layer_name][cell_name] = {
                "n_neurons": n,
                "total_spikes": int(np.sum(spike_counts[mask])),
                "mean_spikes_per_neuron": float(np.mean(spike_counts[mask])),
                "min_spikes_per_neuron": int(np.min(spike_counts[mask])),
                "max_spikes_per_neuron": int(np.max(spike_counts[mask])),
            }

    return {
        "status": "ACCEPT_CANDIDATE" if int(np.min(spike_counts)) >= 1 else "REVISE",
        "truth_status": "truth_safe_unverified",
        "have_jbiophysic_imports": HAVE_JBIOPHYSIC,
        "config": asdict(cfg),
        "n_neurons": int(len(pop.neuron_id)),
        "counts": {
            "E": int(np.sum(pop.cell_type == "E")),
            "PV": int(np.sum(pop.cell_type == "PV")),
            "SST": int(np.sum(pop.cell_type == "SST")),
            "superficial": int(np.sum(pop.layer == "superficial")),
            "mid": int(np.sum(pop.layer == "mid")),
            "deep": int(np.sum(pop.layer == "deep")),
        },
        "spike_floor": {
            "min_spikes_per_neuron": int(np.min(spike_counts)),
            "silent_neurons": int(np.sum(spike_counts == 0)),
            "calibration_attempt": int(result["calibration_attempt"]),
        },
        "layer_summary": layer_summary,
        "units": {
            "time": "ms",
            "dt": "ms",
            "position": "m",
            "voltage": "mV for neuron membrane traces; V for TFNE extracellular phi",
            "izhikevich_current": "native current-like units, not nA",
            "tfne_source_current": "A after explicit spike-to-current calibration",
            "tfne_source_density": "A/m^3",
        },
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, default=Path("outputs/tfne_izhikevich_laminar_ei100"))
    p.add_argument("--seed", type=int, default=17)
    p.add_argument("--t-ms", type=float, default=1000.0)
    p.add_argument("--dt-ms", type=float, default=0.1)
    p.add_argument("--grid-h-um", type=float, default=25.0)
    p.add_argument("--poisson-steps", type=int, default=120)
    p.add_argument("--field-stride-ms", type=float, default=10.0)
    p.add_argument("--calibration-attempts", type=int, default=8)
    p.add_argument("--silent-bias-increment", type=float, default=1.5)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = SimConfig(
        seed=args.seed,
        t_ms=args.t_ms,
        dt_ms=args.dt_ms,
        grid_h_m=args.grid_h_um * 1e-6,
        poisson_steps=args.poisson_steps,
        field_stride_ms=args.field_stride_ms,
        calibration_attempts=args.calibration_attempts,
        silent_bias_increment=args.silent_bias_increment,
    )
    if cfg.dt_ms <= 0 or cfg.t_ms <= 0:
        raise ValueError("time parameters must be positive")

    out = args.out
    out.mkdir(parents=True, exist_ok=True)

    pop = build_population(cfg)
    weights = build_all_to_all_uniform_weights(pop, cfg)
    result = calibrate_until_all_fire(pop, weights, cfg)
    spikes = result["spikes"]
    voltage_mV = result["voltage_mV"]
    spike_counts = spikes.sum(axis=0).astype(np.int64)

    if int(np.min(spike_counts)) < 1:
        raise AssertionError("Every neuron must fire at least once; spike floor failed.")
    if not np.all(np.isfinite(voltage_mV)):
        raise FloatingPointError("voltage_mV contains NaN/Inf")

    grid = make_cylindrical_tfne_grid(cfg)
    kernels = precompute_source_kernels(pop, grid, cfg)
    fields = compute_tfne_field_snapshots(pop, spikes, grid, kernels, cfg)

    for key in ("q_A_per_m3", "phi_V", "source_current_A"):
        if not np.all(np.isfinite(fields[key])):
            raise FloatingPointError(f"{key} contains NaN/Inf")

    # Save arrays.
    np.savez_compressed(out / "spikes_and_voltage.npz", spikes=spikes, voltage_mV=voltage_mV)
    np.savez_compressed(out / "weights_post_pre.npz", weights_post_pre=weights)
    np.savez_compressed(
        out / "tfne_grid.npz",
        x_m=grid.x_m,
        y_m=grid.y_m,
        z_m=grid.z_m,
        active_mask=grid.active_mask,
        voxel_volume_m3=np.asarray(grid.voxel_volume_m3),
        h_m=np.asarray(grid.h_m),
    )
    np.savez_compressed(out / "tfne_field_snapshots.npz", **fields)

    write_population_csv(out / "neuron_table.csv", pop, spike_counts)
    write_rate_csv(out / "population_rates_1ms.csv", population_rate_table(pop, spikes, cfg))

    summary = summarize(pop, spikes, result, cfg)
    with (out / "summary.json").open("w") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))
    print(f"\nWrote outputs to: {out.resolve()}")


if __name__ == "__main__":
    main()
