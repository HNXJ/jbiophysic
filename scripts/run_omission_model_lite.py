#!/usr/bin/env python
"""Deterministic omission model-lite evidence runner.

This is the first executable R5 surface: a two/three-area Izhikevich scaffold
that preserves -500 to +1000 ms event windows, exports spike/rate/spectral/
synchrony/source-proxy diagnostics, and refuses mechanism or calibrated field
claims. It is intentionally lightweight so simulation work can proceed from a
stable evidence contract.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from jbiophysic.io.manifests import hash_assets, write_json_manifest
from jbiophysic.models.laminar_oddball import (
    build_three_area_cortex,
    simulate_laminar_izhikevich,
    summarize_simulation,
)
from jbiophysic.tfne.operator_status import operator_status_by_symbol_json


def _load_cfg(path: str | Path) -> dict[str, Any]:
    return yaml.safe_load(Path(path).read_text())


def _window_slices(dt_ms: float, duration_ms: float) -> dict[str, slice]:
    # Simulation index 0 corresponds to -500 ms; index 500 ms corresponds to event time 0.
    offset_ms = 500.0
    total_steps = int(round(duration_ms / dt_ms))

    def sl(a: float, b: float) -> slice:
        i0 = max(0, int(round((a + offset_ms) / dt_ms)))
        i1 = min(total_steps, int(round((b + offset_ms) / dt_ms)))
        return slice(i0, i1)

    return {
        "baseline": sl(-500.0, 0.0),
        "event": sl(0.0, 500.0),
        "post": sl(500.0, 1000.0),
    }


def _make_drive(cortex, *, condition: str, bottom_up: bool, top_down: bool | str) -> np.ndarray:
    steps = int(round(cortex.duration_ms / cortex.dt_ms))
    drive = np.full((steps, cortex.n_neurons), 2.0, dtype=np.float32)
    wins = _window_slices(cortex.dt_ms, cortex.duration_ms)
    event = wins["event"]
    post = wins["post"]
    low_l4_e = np.flatnonzero(
        (cortex.area_name_array == "low")
        & (cortex.layer_name_array == "middle")
        & (cortex.cell_type_names == "E")
    )
    high_deep_e = np.flatnonzero(
        (cortex.area_name_array == "high")
        & (cortex.layer_name_array == "deep")
        & (cortex.cell_type_names == "E")
    )
    high_sst = np.flatnonzero(
        (cortex.area_name_array == "high") & (cortex.cell_type_names == "SST")
    )
    low_deep_sst = np.flatnonzero(
        (cortex.area_name_array == "low")
        & (cortex.layer_name_array == "deep")
        & (cortex.cell_type_names == "SST")
    )
    if bottom_up:
        gain = 9.5 if condition == "post_omission" else 7.0
        drive[event, low_l4_e] += gain
    if top_down:
        drive[event, high_deep_e] += 7.5
        drive[event, high_sst] += 2.5
        drive[event, low_deep_sst] += 3.0
    if condition == "post_omission":
        drive[post, low_l4_e] += 8.0
        drive[post, high_deep_e] += 2.0
    return drive


def _rate(spikes: np.ndarray, dt_ms: float, idx: np.ndarray, window: slice) -> float:
    duration_s = max(1e-12, (window.stop - window.start) * dt_ms / 1000.0)
    if idx.size == 0:
        return 0.0
    return float(spikes[window][:, idx].sum() / idx.size / duration_s)


def _bandpowers(trace: np.ndarray, dt_ms: float, bands: dict[str, list[float]]) -> dict[str, float]:
    x = np.asarray(trace, dtype=float)
    x = x - float(np.mean(x))
    if x.size < 4:
        return {k: 0.0 for k in bands}
    freqs = np.fft.rfftfreq(x.size, d=dt_ms / 1000.0)
    power = np.abs(np.fft.rfft(x)) ** 2 / max(1, x.size)
    out = {}
    for name, (lo, hi) in bands.items():
        mask = (freqs >= float(lo)) & (freqs <= float(hi))
        out[name] = float(np.mean(power[mask])) if np.any(mask) else 0.0
    return out


def _population_trace(spikes: np.ndarray, idx: np.ndarray, dt_ms: float) -> np.ndarray:
    if idx.size == 0:
        return np.zeros(spikes.shape[0], dtype=float)
    return spikes[:, idx].mean(axis=1).astype(float) * (1000.0 / dt_ms)


def _synchrony_proxy(spikes: np.ndarray, window: slice) -> float:
    x = spikes[window].astype(float)
    if x.shape[0] < 2 or x.shape[1] < 2:
        return 0.0
    active = x.sum(axis=0) > 0
    if active.sum() < 2:
        return 0.0
    corr = np.corrcoef(x[:, active].T)
    if corr.ndim != 2:
        return 0.0
    tri = corr[np.triu_indices_from(corr, 1)]
    tri = tri[np.isfinite(tri)]
    return float(np.mean(tri)) if tri.size else 0.0


def _condition_flags(cfg: dict[str, Any]) -> dict[str, tuple[bool, bool | str]]:
    matrix = cfg.get("condition_matrix", {})
    out: dict[str, tuple[bool, bool | str]] = {}
    for name in [
        "baseline",
        "unexpected_sensory",
        "predicted_standard",
        "omission",
        "post_omission",
    ]:
        row = matrix.get(name, {})
        out[name] = (bool(row.get("bottom_up", False)), row.get("top_down", False))
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--seeds", nargs="*", type=int, default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args(argv)

    cfg = _load_cfg(args.config)
    seeds = args.seeds if args.seeds else list(cfg.get("seeds", [cfg.get("seed", 0)]))
    if args.smoke:
        seeds = seeds[:1]
        cfg["n_neurons"] = max(180, min(int(cfg.get("n_neurons", 180)), 180))
        cfg["dt_ms"] = max(float(cfg.get("dt_ms", 1.0)), 1.0)
        cfg["duration_ms"] = 1500.0

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    bands = cfg.get("bands", {"alpha_beta": [8.0, 30.0], "gamma": [35.0, 90.0]})
    all_conditions = _condition_flags(cfg)

    condition_rows = []
    spike_rows = []
    band_rows = []
    sync_rows = []
    param_rows = []
    source_rows = []
    results_by_seed_condition = {}

    for seed in seeds:
        cortex = build_three_area_cortex(
            n_neurons=int(cfg.get("n_neurons", 60)),
            dt_ms=float(cfg.get("dt_ms", 1.0)),
            duration_ms=float(cfg.get("duration_ms", 1500.0)),
            seed=int(seed),
        )
        wins = _window_slices(cortex.dt_ms, cortex.duration_ms)
        area_labels = cortex.area_name_array
        ctype_labels = cortex.cell_type_names
        for condition, (bottom_up, top_down) in all_conditions.items():
            drive = _make_drive(cortex, condition=condition, bottom_up=bottom_up, top_down=top_down)
            result = simulate_laminar_izhikevich(
                cortex,
                drive,
                seed=int(seed) + 101 * len(condition),
                noise_sd=0.45,
                plasticity_enabled=False,
            )
            results_by_seed_condition[(seed, condition)] = result
            spikes = np.asarray(result["spikes"], dtype=bool)
            summary = summarize_simulation(result, cortex.dt_ms)
            param_rows.append({"seed": seed, "condition": condition, **summary})
            source_rows.append(
                {
                    "seed": seed,
                    "condition": condition,
                    "source_decomposition": "proxy_no_field_solve",
                    "source_projection_mode": "spike_source_proxy_no_field_solve",
                    "source_calibration_status": cfg.get(
                        "source_calibration_status", "uncalibrated_spike_only"
                    ),
                    "kernel_norm_max_abs_error": None,
                    "integrated_current_error_max_abs": None,
                    "neumann_compatibility_max_abs": None,
                    "boundary_condition": None,
                    "gauge_type": None,
                    "solver_status": "not_solved",
                    "solver_residual_l2_relative": None,
                    "conductivity_min_eigenvalue": None,
                    "conductivity_symmetric_error": None,
                    "CSD_sign_convention": "positive_equals_extracellular_source_if_field_solved",
                    "finite_phi_e": None,
                    "finite_J_e": None,
                    "finite_CSD": None,
                    "finite_proxy_source": bool(np.isfinite(spikes.astype(float)).all()),
                }
            )
            for window_name, window in wins.items():
                for area in cortex.area_names:
                    aidx = np.flatnonzero(area_labels == area)
                    condition_rows.append(
                        {
                            "seed": seed,
                            "condition": condition,
                            "window": window_name,
                            "area": area,
                            "firing_rate_hz": _rate(spikes, cortex.dt_ms, aidx, window),
                        }
                    )
                    trace = (
                        _population_trace(spikes[window], np.arange(aidx.size), cortex.dt_ms)
                        if False
                        else _population_trace(spikes, aidx, cortex.dt_ms)[window]
                    )
                    powers = _bandpowers(trace, cortex.dt_ms, bands)
                    band_rows.append(
                        {
                            "seed": seed,
                            "condition": condition,
                            "window": window_name,
                            "area": area,
                            **powers,
                        }
                    )
                for cell_type in cortex.cell_types:
                    cidx = np.flatnonzero(ctype_labels == cell_type)
                    spike_rows.append(
                        {
                            "seed": seed,
                            "condition": condition,
                            "window": window_name,
                            "cell_type": cell_type,
                            "firing_rate_hz": _rate(spikes, cortex.dt_ms, cidx, window),
                            "active_fraction": float(
                                np.mean(spikes[window][:, cidx].sum(axis=0) > 0)
                            )
                            if cidx.size
                            else 0.0,
                        }
                    )
                sync_rows.append(
                    {
                        "seed": seed,
                        "condition": condition,
                        "window": window_name,
                        "synchrony_metric": "mean_pairwise_correlation_proxy",
                        "synchrony_value": _synchrony_proxy(spikes, window),
                    }
                )

    condition_df = pd.DataFrame(condition_rows)
    spike_df = pd.DataFrame(spike_rows)
    band_df = pd.DataFrame(band_rows)
    sync_df = pd.DataFrame(sync_rows)
    param_df = pd.DataFrame(param_rows)
    source_df = pd.DataFrame(source_rows)

    condition_df.to_csv(out / "condition_metrics.csv", index=False)
    spike_df.to_csv(out / "spike_diagnostics.csv", index=False)
    source_df.to_csv(out / "field_invariants.csv", index=False)
    band_df.to_csv(out / "bandpower_by_condition.csv", index=False)
    sync_df.to_csv(out / "synchrony_diagnostics.csv", index=False)
    param_df.to_csv(out / "parameter_bounds.csv", index=False)

    null_df = pd.DataFrame(
        [
            {
                "null_type": name,
                "status": "declared_for_serious_run",
                "claim": "not_executed_in_smoke",
            }
            for name in [
                "no_top_down_feedback",
                "unstructured_random_feedback",
                "bottom_up_residual_drive_control",
                "phase_randomized",
                "matched_power_high_synchrony_invalid",
            ]
        ]
    )
    null_df.to_csv(out / "null_metrics.csv", index=False)
    ablation_df = pd.DataFrame(
        [
            {
                "ablation": name,
                "status": "declared_for_serious_run",
                "claim": "not_executed_in_smoke",
            }
            for name in [
                "no_feedback",
                "PV_only_feedback",
                "SST_only_feedback",
                "VIP_route",
                "local_adaptation_only",
            ]
        ]
    )
    ablation_df.to_csv(out / "ablation_metrics.csv", index=False)

    # Smoke decision is conservative.
    # This runner closes the contract; empirical acceptance needs multi-seed controls.
    manifest = {
        "truth_mode": cfg.get("truth_mode", "truth_safe_unverified"),
        "claim_level": cfg.get("claim_level", "empirical_convergence_scaffold"),
        "model": "omission_model_lite",
        "seeds": seeds,
        "windows_ms": cfg.get("windows_ms"),
        "condition_matrix": cfg.get("condition_matrix"),
        "source_calibration_status": cfg.get(
            "source_calibration_status", "uncalibrated_spike_only"
        ),
        "field_mode": cfg.get("field_mode", "proxy_no_field_solve"),
        "operator_status_by_symbol": operator_status_by_symbol_json(),
        "acceptance_decision": cfg.get("acceptance", {}).get(
            "decision_for_smoke", "REVISE_BEFORE_EMPIRICAL_CLAIM"
        ),
        "interpretation": (
            "R5 executable surface; smoke output is not biological proof or mechanism acceptance."
        ),
        "required_outputs": [
            "manifest.json",
            "condition_metrics.csv",
            "spike_diagnostics.csv",
            "field_invariants.csv",
            "bandpower_by_condition.csv",
            "synchrony_diagnostics.csv",
            "ablation_metrics.csv",
            "null_metrics.csv",
            "parameter_bounds.csv",
            "asset_hashes.json",
        ],
    }
    asset_paths = [
        p
        for p in out.rglob("*")
        if p.is_file() and p.name not in {"asset_hashes.json", "manifest.json"}
    ]
    asset_hashes = hash_assets(asset_paths)
    manifest["asset_hashes"] = asset_hashes
    write_json_manifest(out / "asset_hashes.json", asset_hashes)
    write_json_manifest(out / "manifest.json", manifest)
    print(
        json.dumps(
            {"status": "ok", "out": str(out), "decision": manifest["acceptance_decision"]}, indent=2
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
