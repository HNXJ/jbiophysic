"""Manifest and report harmonization with strict JSON safety.

Core functions:
- get_installed_jaxfne_version() - runtime version resolution
- json_safe() - safe conversion of NumPy/pandas/Path to JSON types
- write_manifest() - strict JSON serialization with NaN/Inf rejection
- harmonize_jaxfne_output() - convert raw jaxfne output to harmonized format
"""

from __future__ import annotations

import json
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any


def get_installed_jaxfne_version() -> str:
    """Resolve installed jaxfne version at runtime.

    Returns
    -------
    str
        Installed jaxfne version or "unknown" if not found.
    """
    try:
        return version("jaxfne")
    except PackageNotFoundError:
        return "unknown"


def json_safe(obj: Any) -> Any:
    """Convert Python/NumPy/pandas objects to JSON-safe types.

    Handles:
    - dict: recursively convert to safe types
    - list/tuple: convert to list
    - NumPy arrays: convert to Python lists
    - NumPy scalars: convert to Python scalars
    - pandas DataFrames: convert to list of dicts
    - pathlib.Path: convert to string
    - NaN/Inf: converted to None
    - Others: return as-is (will fail json.dumps if not serializable)

    Parameters
    ----------
    obj : Any
        Object to convert.

    Returns
    -------
    Any
        JSON-safe version of obj.
    """
    # Try NumPy first
    try:
        import numpy as np

        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.integer, np.floating)):
            val = obj.item()
            # Convert NaN/Inf to None
            if (
                isinstance(val, float)
                and (not (-1e308 < val < 1e308) or val != val)  # NaN or Inf check
            ):
                return None
            return val
    except ImportError:
        pass

    # Try pandas
    try:
        import pandas as pd

        if isinstance(obj, pd.DataFrame):
            return obj.to_dict(orient="records")
        if isinstance(obj, (pd.Series, pd.Index)):
            return obj.tolist()
    except ImportError:
        pass

    if isinstance(obj, dict):
        return {str(k): json_safe(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [json_safe(item) for item in obj]
    elif isinstance(obj, Path):
        return str(obj)
    elif isinstance(obj, float):
        # Check for NaN/Inf
        if not (-1e308 < obj < 1e308) or obj != obj:  # NaN or Inf check
            return None
        return obj
    else:
        return obj


def write_manifest(
    manifest: dict[str, Any],
    output_path: str,
    allow_nan: bool = False,
) -> str:
    """Write manifest to JSON with strict safety checks.

    Parameters
    ----------
    manifest : Dict[str, Any]
        Manifest dict to serialize.
    output_path : str
        Path to write manifest JSON.
    allow_nan : bool, optional
        If False (default), reject NaN/Inf. If True, allow them.

    Returns
    -------
    str
        Path to written manifest file.

    Raises
    ------
    ValueError
        If allow_nan=False and NaN/Inf detected.
    """
    # Convert to JSON-safe types
    safe_manifest = json_safe(manifest)

    # If strict, check for remaining NaN/Inf before serialization
    if not allow_nan:

        def check_nans(obj: Any, path: str = "root") -> None:
            if isinstance(obj, dict):
                for k, v in obj.items():
                    check_nans(v, f"{path}.{k}")
            elif isinstance(obj, (list, tuple)):
                for i, item in enumerate(obj):
                    check_nans(item, f"{path}[{i}]")
            elif (
                isinstance(obj, float)
                and (not (-1e308 < obj < 1e308) or obj != obj)  # NaN/Inf check
            ):
                raise ValueError(f"NaN/Inf detected at {path}: {obj}")

        check_nans(safe_manifest)

    # Create parent directories
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    # Write with strict JSON (no allow_nan unless explicitly requested)
    with open(output_path, "w") as f:
        json.dump(safe_manifest, f, indent=2, allow_nan=allow_nan)

    return output_path


def harmonize_jaxfne_output(
    jaxfne_result: dict[str, Any],
    manifest: dict[str, Any],
) -> dict[str, Any]:
    """Convert raw jaxfne output to harmonized format.

    Parameters
    ----------
    jaxfne_result : Dict[str, Any]
        Raw jaxfne engine output.
    manifest : Dict[str, Any]
        Original manifest for context.

    Returns
    -------
    Dict[str, Any]
        Harmonized output dict.
    """
    return {
        "v_trace": jaxfne_result.get("v_trace"),
        "spike_times": jaxfne_result.get("spike_times", []),
        "lfp": jaxfne_result.get("lfp"),
        "csd": jaxfne_result.get("csd"),
        "metadata": {
            "source_type": manifest.get("source_type"),
            "source_scale": manifest.get("source_scale"),
            "duration_ms": manifest.get("parameters", {}).get("duration_ms"),
            "dt_ms": manifest.get("parameters", {}).get("dt_ms"),
        },
    }
