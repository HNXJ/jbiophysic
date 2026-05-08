"""JSON manifest IO helpers."""

from __future__ import annotations

import json
from pathlib import Path


def write_json_manifest(path: str | Path, payload: dict[str, object]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str, sort_keys=True) + "\n")


def read_json_manifest(path: str | Path) -> dict[str, object]:
    return json.loads(Path(path).read_text())


def create_run_manifest(
    seed: int,
    dt_ms: float,
    duration_ms: float,
    model_family: str = "izhikevich",
    current_unit: str = "izhikevich_model_unit",
    nan_inf_status: str = "pass",
    bounds_status: str = "pass",
) -> dict[str, object]:
    """Create a standard run manifest for tutorial provenance."""
    return {
        "truth_status": "exploratory_not_biological_truth",
        "seed": seed,
        "dt_ms": dt_ms,
        "simulation_time_ms": duration_ms,
        "model_family": model_family,
        "current_unit": current_unit,
        "tfne_source_calibration": None,
        "nan_inf_status": nan_inf_status,
        "bounds_status": bounds_status,
    }
