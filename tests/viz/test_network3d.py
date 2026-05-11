from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest

from jbiophysic.viz.network3d import (
    build_laminar_population_anatomy,
    build_two_cortex_laminar_anatomy,
    visualize_network_3d,
)


def _plotly_available() -> bool:
    try:
        import plotly.graph_objects  # noqa: F401
    except Exception:
        return False
    return True


def _minimum_pairwise_distance(positions: np.ndarray) -> float:
    best = np.inf
    for i in range(len(positions)):
        d = np.linalg.norm(positions[i + 1 :] - positions[i], axis=1)
        if d.size:
            best = min(best, float(np.min(d)))
    return best


def test_import_smoke():
    from jbiophysic.viz.network3d import visualize_network_3d as imported

    assert imported is visualize_network_3d


@pytest.mark.skipif(not _plotly_available(), reason="plotly unavailable")
def test_1d_input_maps_to_finite_3d_coordinates():
    fig, rows = visualize_network_3d(
        {"positions_m": np.asarray([0.0, 1.0e-6, 2.0e-6])},
        return_node_table=True,
    )
    assert fig is not None
    xyz = np.asarray([[row["x_m"], row["y_m"], row["z_m"]] for row in rows], dtype=float)
    assert xyz.shape == (3, 3)
    assert np.all(np.isfinite(xyz))
    assert np.allclose(xyz[:, 1], 0.0)
    assert np.allclose(xyz[:, 2], 0.0)


@pytest.mark.skipif(not _plotly_available(), reason="plotly unavailable")
def test_2d_input_maps_to_finite_3d_coordinates():
    fig, rows = visualize_network_3d(
        {"positions_m": np.asarray([[0.0, 0.0], [1.0e-6, 2.0e-6]])},
        return_node_table=True,
    )
    assert fig is not None
    xyz = np.asarray([[row["x_m"], row["y_m"], row["z_m"]] for row in rows], dtype=float)
    assert xyz.shape == (2, 3)
    assert np.all(np.isfinite(xyz))
    assert np.allclose(xyz[:, 2], 0.0)


@pytest.mark.skipif(not _plotly_available(), reason="plotly unavailable")
def test_3d_input_preserves_coordinates_without_jitter():
    positions = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [1.0e-6, 2.0e-6, 3.0e-6],
            [4.0e-6, 5.0e-6, 6.0e-6],
        ]
    )
    _, rows = visualize_network_3d(
        {"positions_m": positions},
        jitter_duplicates=False,
        return_node_table=True,
    )
    xyz = np.asarray([[row["x_m"], row["y_m"], row["z_m"]] for row in rows], dtype=float)
    assert np.allclose(xyz, positions)


@dataclass(frozen=True)
class StarterStylePopulation:
    neuron_id: np.ndarray
    layer: np.ndarray
    cell_type: np.ndarray
    x_m: np.ndarray
    y_m: np.ndarray
    z_m: np.ndarray


@pytest.mark.skipif(not _plotly_available(), reason="plotly unavailable")
def test_starter_style_population_object_is_accepted():
    pop = StarterStylePopulation(
        neuron_id=np.arange(3),
        layer=np.asarray(["superficial", "mid", "deep"]),
        cell_type=np.asarray(["E", "PV", "SST"]),
        x_m=np.asarray([0.0, 10.0e-6, 20.0e-6]),
        y_m=np.asarray([0.0, 0.0, 0.0]),
        z_m=np.asarray([10.0e-6, 350.0e-6, 800.0e-6]),
    )
    _, rows = visualize_network_3d(pop, return_node_table=True)
    assert len(rows) == 3
    assert {row["cell_type"] for row in rows} == {"E", "PV", "SST"}


@pytest.mark.skipif(not _plotly_available(), reason="plotly unavailable")
def test_duplicate_coordinates_are_jittered_deterministically():
    network = {
        "positions_m": np.asarray(
            [
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [1.0e-6, 0.0, 0.0],
            ]
        )
    }
    _, rows_a = visualize_network_3d(
        network,
        min_separation_m=2.0e-6,
        seed=123,
        return_node_table=True,
    )
    _, rows_b = visualize_network_3d(
        network,
        min_separation_m=2.0e-6,
        seed=123,
        return_node_table=True,
    )
    xyz_a = np.asarray([[row["x_m"], row["y_m"], row["z_m"]] for row in rows_a], dtype=float)
    xyz_b = np.asarray([[row["x_m"], row["y_m"], row["z_m"]] for row in rows_b], dtype=float)
    assert np.allclose(xyz_a, xyz_b)
    assert any(row["jittered"] for row in rows_a)
    assert _minimum_pairwise_distance(xyz_a) >= 2.0e-6 - 1.0e-12


def test_generated_laminar_population_has_expected_counts_and_no_overlap():
    network = build_laminar_population_anatomy(seed=17, min_separation_m=4.0e-6)
    assert len(network["positions_m"]) == 100
    assert set(network["cell_type"]) == {"E", "PV", "SST"}
    assert set(network["layer"]) == {"superficial", "mid", "deep"}
    assert _minimum_pairwise_distance(np.asarray(network["positions_m"], dtype=float)) >= 4.0e-6


@pytest.mark.skipif(not _plotly_available(), reason="plotly unavailable")
def test_generated_two_column_demo_has_expected_areas_cell_types_layers_and_no_overlap():
    network = build_two_cortex_laminar_anatomy(seed=17, min_separation_m=4.0e-6)
    assert set(network["area"]) == {"lower-cortex", "higher-cortex"}
    assert set(network["cell_type"]) == {"E", "PV", "SST"}
    assert set(network["layer"]) == {"superficial", "mid", "deep"}
    assert len(network["positions_m"]) == 200
    assert _minimum_pairwise_distance(np.asarray(network["positions_m"], dtype=float)) >= 4.0e-6

    _, rows = visualize_network_3d(
        network,
        min_separation_m=4.0e-6,
        return_node_table=True,
    )
    assert len(rows) == 200
    assert set(row["area"] for row in rows) == {"lower-cortex", "higher-cortex"}


@pytest.mark.skipif(not _plotly_available(), reason="plotly unavailable")
def test_html_write_smoke(tmp_path: Path):
    out = tmp_path / "network3d.html"
    network = build_two_cortex_laminar_anatomy(seed=17, min_separation_m=4.0e-6)
    visualize_network_3d(network, output_html=out, min_separation_m=4.0e-6)
    assert out.exists()
    text = out.read_text(encoding="utf-8")
    assert "Plotly.newPlot" in text
