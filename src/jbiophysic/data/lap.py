"""Load LAP laminar cell-count MATLAB data.

The LAP parser is intentionally conservative: it extracts marker/layer counts
and preserves them as floating-point processed counts. Integer model allocation
is handled downstream by model builders, not by this parser.
"""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy.io import loadmat

EXPECTED_MARKERS: tuple[str, ...] = ("PV", "CB", "CR", "NG")
_OPTIONAL_AREA_FIELDS = ("cellPercentages",)


@dataclass(frozen=True)
class LAPCountRow:
    area: str
    marker: str
    layer_index: int
    layer_label: str
    count: float


def load_lap_mat(path: str | Path) -> Any:
    """Load a MATLAB v5 LAP `.mat` file and return its `cellularData` root.

    Parameters
    ----------
    path:
        Path to the LAP MATLAB file.
    """
    mat_path = Path(path)
    data = loadmat(mat_path, squeeze_me=True, struct_as_record=False)
    if "cellularData" not in data:
        raise KeyError("LAP .mat file must contain root variable 'cellularData'")
    return data["cellularData"]


def _field_names(obj: Any) -> list[str]:
    names = getattr(obj, "_fieldnames", None)
    if names is None:
        if isinstance(obj, dict):
            names = [str(k) for k in obj]
        else:
            names = [name for name in dir(obj) if not name.startswith("_")]
    return [str(name) for name in names if not str(name).startswith("_")]


def _get_field(obj: Any, name: str) -> Any:
    if isinstance(obj, dict):
        return obj[name]
    return getattr(obj, name)


def _has_field(obj: Any, name: str) -> bool:
    if isinstance(obj, dict):
        return name in obj
    return hasattr(obj, name)


def _as_float_vector(value: Any) -> np.ndarray:
    arr = np.asarray(value, dtype=float)
    arr = np.atleast_1d(arr).reshape(-1)
    if not np.all(np.isfinite(arr)):
        raise ValueError("layerCounts contains NaN or Inf")
    return arr


def extract_lap_layer_counts(path: str | Path) -> list[LAPCountRow]:
    """Extract area / marker / layer counts from LAP `cellularData`.

    Counts are returned as float values because LAP `layerCounts` are processed
    and can be fractional. Layer labels are generated as L1..Ln, so PMD-like
    three-layer areas remain three-layer areas.
    """
    cellular_data = load_lap_mat(path)
    rows: list[LAPCountRow] = []

    for area in sorted(_field_names(cellular_data)):
        area_obj = _get_field(cellular_data, area)
        for marker in EXPECTED_MARKERS:
            if not _has_field(area_obj, marker):
                continue
            marker_obj = _get_field(area_obj, marker)
            if not _has_field(marker_obj, "layerCounts"):
                continue
            counts = _as_float_vector(_get_field(marker_obj, "layerCounts"))
            for layer_zero_idx, count in enumerate(counts):
                rows.append(
                    LAPCountRow(
                        area=area,
                        marker=marker,
                        layer_index=layer_zero_idx + 1,
                        layer_label=f"L{layer_zero_idx + 1}",
                        count=float(count),
                    )
                )
    if not rows:
        raise ValueError("No LAP layerCounts were found under cellularData")
    return rows


def lap_counts_to_long_table(path: str | Path) -> list[dict[str, object]]:
    """Return LAP layer counts as CSV/JSON-friendly dictionaries."""
    return [asdict(row) for row in extract_lap_layer_counts(path)]


def summarize_lap_counts(path: str | Path) -> dict[str, object]:
    """Summarize LAP areas, markers, layer counts, and optional-field warnings."""
    cellular_data = load_lap_mat(path)
    rows = extract_lap_layer_counts(path)
    areas = sorted({row.area for row in rows})

    markers_by_area: dict[str, list[str]] = {}
    layers_by_area_marker: dict[str, dict[str, int]] = {}
    counts_by_area: dict[str, float] = {}
    counts_by_marker: dict[str, float] = {}
    warnings: list[str] = []

    for area in areas:
        area_obj = _get_field(cellular_data, area)
        markers = [marker for marker in EXPECTED_MARKERS if _has_field(area_obj, marker)]
        markers_by_area[area] = markers
        missing_markers = [marker for marker in EXPECTED_MARKERS if marker not in markers]
        for marker in missing_markers:
            warnings.append(f"{area} missing marker {marker}")
        for optional in _OPTIONAL_AREA_FIELDS:
            if not _has_field(area_obj, optional):
                warnings.append(f"{area} missing optional field {optional}")

        layers_by_area_marker[area] = {}
        area_rows = [row for row in rows if row.area == area]
        counts_by_area[area] = float(sum(row.count for row in area_rows))
        layer_counts = sorted({row.layer_index for row in area_rows})
        if len(layer_counts) != 6:
            warnings.append(f"{area} has {len(layer_counts)} layers in layerCounts")
        for marker in markers:
            marker_rows = [row for row in area_rows if row.marker == marker]
            layers_by_area_marker[area][marker] = len(marker_rows)

    for marker in EXPECTED_MARKERS:
        counts_by_marker[marker] = float(sum(row.count for row in rows if row.marker == marker))

    return {
        "mat_file": Path(path).name,
        "root_variable": "cellularData",
        "n_areas": len(areas),
        "areas": areas,
        "expected_markers": list(EXPECTED_MARKERS),
        "markers_by_area": markers_by_area,
        "layers_by_area_marker": layers_by_area_marker,
        "counts_by_area": counts_by_area,
        "counts_by_marker": counts_by_marker,
        "n_rows": len(rows),
        "warnings": sorted(set(warnings)),
    }


def write_lap_counts_csv(path: str | Path, out_csv: str | Path) -> None:
    """Write long-form LAP layer counts to CSV."""
    rows = lap_counts_to_long_table(path)
    out_path = Path(out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["area", "marker", "layer_index", "layer_label", "count"]
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
