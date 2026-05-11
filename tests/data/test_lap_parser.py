from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.io import savemat

from jbiophysic.data.lap import extract_lap_layer_counts, load_lap_mat, summarize_lap_counts


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
            "CB": marker([1, 0, 2, 0, 4, 0]),
            "CR": marker([2, 2, 2, 2, 2, 2]),
            "NG": marker([10, 20, 30, 40, 50, 60]),
            "cellPercentages": np.asarray([1, 2, 3, 4], dtype=float),
        },
        "PMD": {
            "PV": marker([1, 2, 3]),
            "CB": marker([0, 2, 0]),
            "CR": marker([3, 3, 3]),
            "NG": marker([10, 20, 30]),
            "cellPercentages": np.asarray([1, 2, 3, 4], dtype=float),
        },
        "DP": {
            "PV": marker([1, 1, 1, 1, 1, 1]),
            "CB": marker([0, 0, 1, 0, 0, 0]),
            "CR": marker([2, 0, 0, 0, 0, 0]),
            "NG": marker([5, 5, 5, 5, 5, 5]),
            # intentionally no cellPercentages
        },
    }
    savemat(path, {"cellularData": cellular_data})
    return path


def test_load_lap_mat_root(tmp_path: Path):
    mat_path = write_synthetic_lap_mat(tmp_path / "lap.mat")
    root = load_lap_mat(mat_path)
    assert hasattr(root, "V1")
    assert hasattr(root, "PMD")
    assert hasattr(root, "DP")


def test_extract_lap_counts_preserves_float_counts_and_layers(tmp_path: Path):
    mat_path = write_synthetic_lap_mat(tmp_path / "lap.mat")
    rows = extract_lap_layer_counts(mat_path)
    assert rows
    assert all(isinstance(row.count, float) for row in rows)

    pmd_rows = [row for row in rows if row.area == "PMD"]
    assert {row.layer_label for row in pmd_rows} == {"L1", "L2", "L3"}
    assert len([row for row in pmd_rows if row.marker == "PV"]) == 3

    dp_rows = [row for row in rows if row.area == "DP"]
    assert len([row for row in dp_rows if row.marker == "NG"]) == 6


def test_summary_accepts_missing_cell_percentages_and_warns(tmp_path: Path):
    mat_path = write_synthetic_lap_mat(tmp_path / "lap.mat")
    summary = summarize_lap_counts(mat_path)
    assert summary["n_areas"] == 3
    warnings = summary["warnings"]
    assert any("DP missing optional field cellPercentages" in warning for warning in warnings)
    assert any("PMD has 3 layers" in warning for warning in warnings)
