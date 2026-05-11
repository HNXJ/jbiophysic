from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.io import savemat

from jbiophysic.data.lap import extract_lap_layer_counts
from jbiophysic.models.lap_izhikevich_baseline import (
    LAPBaselineConfig,
    LAPBaselineConfig,
    build_lap_population,
    build_sparse_baseline_weights,
    run_lap_spontaneous_baseline,
    summarize_lap_baseline,
)


def write_synthetic_lap_mat(path: Path) -> Path:
    def marker(counts):
        return {
            "layerCounts": np.asarray(counts, dtype=float),
            "cellCount": np.asarray(counts, dtype=float),
            "overallDensity": float(np.sum(counts)),
        }

    cellular_data = {
        "V1": {
            "PV": marker([0, 1, 2, 3, 4, 5]),
            "CB": marker([10, 0, 10, 0, 10, 0]),
            "CR": marker([2, 2, 2, 2, 2, 2]),
            "NG": marker([10, 20, 30, 40, 50, 60]),
            "cellPercentages": np.asarray([1, 2, 3, 4], dtype=float),
        },
        "PMD": {
            "PV": marker([1, 2, 3]),
            "CB": marker([0, 10, 0]),
            "CR": marker([3, 3, 3]),
            "NG": marker([10, 20, 30]),
            "cellPercentages": np.asarray([1, 2, 3, 4], dtype=float),
        },
        "DP": {
            "PV": marker([1, 1, 1, 1, 1, 1]),
            "CB": marker([0, 0, 1, 0, 0, 0]),
            "CR": marker([2, 0, 0, 0, 0, 0]),
            "NG": marker([5, 5, 5, 5, 5, 5]),
        },
    }
    savemat(path, {"cellularData": cellular_data})
    return path


def test_allocation_sums_per_area_zero_bins_and_pmd_layers(tmp_path: Path):
    mat_path = write_synthetic_lap_mat(tmp_path / "lap.mat")
    rows = extract_lap_layer_counts(mat_path)
    allocations = allocate_integer_counts_from_lap(rows, neurons_per_area=20)

    for area in {row.area for row in rows}:
        assert sum(int(row["allocated_count"]) for row in allocations if row["area"] == area) == 20

    for row in allocations:
        if float(row["raw_count"]) == 0.0:
            assert int(row["allocated_count"]) == 0

    pmd_layers = {int(row["layer_index"]) for row in allocations if row["area"] == "PMD"}
    assert pmd_layers == {1, 2, 3}


def test_quick_spontaneous_baseline_smoke(tmp_path: Path):
    mat_path = write_synthetic_lap_mat(tmp_path / "lap.mat")
    cfg = LAPBaselineConfig(
        mat_path=mat_path,
        seed=5,
        t_ms=20.0,
        dt_ms=0.5,
        neurons_per_area=20,
        include_areas=("V1", "PMD"),
        mean_in_degree=5,
        min_separation_m=1.0e-6,
    )
    pop = build_lap_population(cfg)
    assert len(pop.neuron_id) == 40
    assert set(pop.area) == {"V1", "PMD"}
    assert set(pop.marker) == {"PV", "CB", "CR", "NG"}
    assert set(pop.layer) <= {"L1", "L2", "L3", "L4", "L5", "L6"}

    weights = build_sparse_baseline_weights(pop, cfg)
    assert weights.shape == (40, 40)
    assert np.all(np.diag(weights) == 0)

    result = run_lap_spontaneous_baseline(pop, weights, cfg)
    assert result["spikes"].shape == (40, 40)
    assert result["voltage_mV"].shape == (40, 40)
    assert np.all(np.isfinite(result["voltage_mV"]))
    assert not bool(np.asarray(result["sensory_input_enabled"]).item())
    assert not bool(np.asarray(result["omission_input_enabled"]).item())
    assert not bool(np.asarray(result["top_down_prediction_enabled"]).item())

    summary = summarize_lap_baseline(pop, result, cfg)
    assert summary["truth_status"] == "truth_safe_unverified"
    assert summary["baseline_mode"] == "spontaneous"
    assert summary["claim_status"]["biological_validation"] is False
    assert summary["inputs_enabled"] == {
        "sensory": False,
        "omission": False,
        "top_down_prediction": False,
    }
