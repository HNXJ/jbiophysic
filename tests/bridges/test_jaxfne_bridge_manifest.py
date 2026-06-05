"""Tests for jaxfne bridge manifest builders.

Focus: Schema completeness, JSON safety, strict validation.
"""

import json
import tempfile
from pathlib import Path

from jbiophysic.bridges.jaxfne import (
    build_ei_network_run,
    build_laminar_proxy_run,
    build_single_neuron_run,
    json_safe,
    validate_manifest_json,
    write_manifest,
)

REQUIRED_MANIFEST_FIELDS = {
    "run_id",
    "run_type",
    "source_type",
    "source_scale",
    "jaxfne_engine_version",
    "jbiophysic_bridge_version",
    "truth_mode",
    "claim_level",
    "physical_amplitude_claim_allowed",
    "source_calibration_status",
    "source_decomposition",
    "field_solver_status",
    "seed",
    "code_version",
    "config_hash",
    "parameters",
    "jaxfne_request",
    "jaxfne_output",
    "harmonized_output",
    "operator_status",
    "outputs",
}


def test_build_single_neuron_run_schema_complete():
    """Test build_single_neuron_run returns all required fields."""
    manifest = build_single_neuron_run(
        cell_type="izhikevich",
        params={"a": 0.02, "b": 0.2, "c": -65.0, "d": 8.0, "I_inj_nA": 10.0},
        stimulus_pattern={"kind": "step"},
        duration_ms=500.0,
        dt_ms=0.1,
    )

    missing = REQUIRED_MANIFEST_FIELDS - set(manifest.keys())
    assert not missing, f"Missing required fields: {missing}"
    assert manifest["run_type"] == "single_neuron"
    assert manifest["source_type"] == "izhikevich"
    assert manifest["source_scale"] == "toy"


def test_build_single_neuron_run_truth_mode():
    """Test builder sets correct truth_mode and claim_level."""
    manifest = build_single_neuron_run(
        cell_type="izhikevich",
        params={"a": 0.02},
        stimulus_pattern={},
        duration_ms=500.0,
        dt_ms=0.1,
    )

    assert manifest["truth_mode"] == "truth_safe_unverified"
    assert manifest["claim_level"] == "computational_scaffold"
    assert manifest["physical_amplitude_claim_allowed"] is False


def test_build_single_neuron_run_parameters():
    """Test builder computes parameters correctly."""
    manifest = build_single_neuron_run(
        cell_type="izhikevich",
        params={},
        stimulus_pattern={},
        duration_ms=500.0,
        dt_ms=0.1,
    )

    params = manifest["parameters"]
    assert params["duration_ms"] == 500.0
    assert params["dt_ms"] == 0.1
    assert params["n_steps"] == 5000


def test_build_ei_network_run_schema_complete():
    """Test build_ei_network_run returns all required fields."""
    manifest = build_ei_network_run(
        n_exc=50,
        n_inh=10,
        connectivity_config={},
        stimulus_config={},
        duration_ms=1000.0,
        dt_ms=0.1,
    )

    missing = REQUIRED_MANIFEST_FIELDS - set(manifest.keys())
    assert not missing, f"Missing required fields: {missing}"
    assert manifest["run_type"] == "ei_network"
    assert manifest["source_type"] == "izhikevich_network"
    assert manifest["source_scale"] == "proxy"


def test_build_laminar_proxy_run_schema_complete():
    """Test build_laminar_proxy_run returns all required fields."""
    manifest = build_laminar_proxy_run(
        laminar_config={},
        source_scale="toy",
        stimulus_pattern={},
        duration_ms=2000.0,
        dt_ms=0.1,
    )

    missing = REQUIRED_MANIFEST_FIELDS - set(manifest.keys())
    assert not missing, f"Missing required fields: {missing}"
    assert manifest["run_type"] == "laminar_proxy"
    assert manifest["source_type"] == "proxy_source"
    assert manifest["source_scale"] == "toy"


def test_laminar_proxy_run_source_scale_mapping():
    """Test laminar proxy maps source_scale to calibration status correctly per doctrine.

    Doctrine mapping (exact):
    - toy → toy_scale_not_empirical
    - proxy → uncalibrated_spike_only (default) or calibrated_proxy (if explicit metadata)
    - calibrated → empirically_calibrated (NOT calibrated_proxy)
    - physical → physical
    """
    for scale, expected_status in [
        ("toy", "toy_scale_not_empirical"),
        ("proxy", "uncalibrated_spike_only"),
        ("calibrated", "empirically_calibrated"),  # CRITICAL: must be empirically_calibrated
        ("physical", "physical"),
    ]:
        manifest = build_laminar_proxy_run(
            laminar_config={},
            source_scale=scale,
            stimulus_pattern={},
            duration_ms=500.0,
            dt_ms=0.1,
        )
        assert manifest["source_calibration_status"] == expected_status, (
            f"Scale {scale} should map to {expected_status}, "
            f"got {manifest['source_calibration_status']}"
        )


def test_manifest_validates():
    """Test manifest passes own validation."""
    manifest = build_single_neuron_run(
        cell_type="izhikevich",
        params={},
        stimulus_pattern={},
        duration_ms=500.0,
        dt_ms=0.1,
    )

    is_valid, errors = validate_manifest_json(manifest)
    assert is_valid, f"Built manifest should validate: {errors}"


def test_write_manifest_json_safe():
    """Test write_manifest converts to JSON-safe types."""
    import numpy as np

    manifest = build_single_neuron_run(
        cell_type="izhikevich",
        params={},
        stimulus_pattern={},
        duration_ms=500.0,
        dt_ms=0.1,
    )

    # Add some NumPy arrays
    manifest["test_array"] = np.array([1.0, 2.0, 3.0])
    manifest["test_scalar"] = np.float32(3.14)

    with tempfile.TemporaryDirectory() as tmpdir:
        path = str(Path(tmpdir) / "test_manifest.json")
        result_path = write_manifest(manifest, path, allow_nan=False)

        # Read back and verify it's valid JSON
        with open(result_path) as f:
            loaded = json.load(f)

        # Arrays should be converted to lists
        assert isinstance(loaded["test_array"], list)
        assert loaded["test_array"] == [1.0, 2.0, 3.0]
        # Scalar should be converted to Python float
        assert isinstance(loaded["test_scalar"], float)


def test_write_manifest_sanitizes_nan_to_none():
    """Test write_manifest converts NaN to None for JSON safety."""
    manifest = {
        "run_id": "test",
        "value": float("nan"),
    }

    with tempfile.TemporaryDirectory() as tmpdir:
        path = str(Path(tmpdir) / "test_manifest.json")
        # json_safe() sanitizes NaN → None, so this should succeed
        result_path = write_manifest(manifest, path, allow_nan=False)

        # Read back and verify NaN was converted to None
        with open(result_path) as f:
            loaded = json.load(f)

        assert loaded["value"] is None, "NaN should be converted to None for JSON safety"


def test_write_manifest_sanitizes_nan_even_with_allow_nan():
    """Test json_safe always sanitizes NaN to None (not just when strict)."""
    manifest = {
        "run_id": "test",
        "value": float("nan"),
    }

    with tempfile.TemporaryDirectory() as tmpdir:
        path = str(Path(tmpdir) / "test_manifest.json")
        # json_safe() sanitizes NaN → None before json.dumps, so
        # allow_nan=True has no effect on NaN values
        result_path = write_manifest(manifest, path, allow_nan=True)

        # Read back
        with open(result_path) as f:
            loaded = json.load(f)

        # json_safe converts NaN to None, so we get null in JSON
        assert loaded["value"] is None, "json_safe should convert NaN to None for compatibility"


def test_json_safe_converts_numpy():
    """Test json_safe converts NumPy types."""
    import numpy as np

    obj = {
        "array": np.array([1, 2, 3]),
        "int": np.int32(42),
        "float": np.float64(3.14),
        "nested": {"inner": np.array([[1, 2], [3, 4]])},
    }

    safe = json_safe(obj)

    assert isinstance(safe["array"], list)
    assert safe["array"] == [1, 2, 3]
    assert isinstance(safe["int"], int)
    assert safe["int"] == 42
    assert isinstance(safe["float"], float)
    assert isinstance(safe["nested"]["inner"], list)


def test_json_safe_converts_path():
    """Test json_safe converts Path to string."""
    from pathlib import Path

    obj = {"path": Path("/tmp/test.txt"), "other": "value"}
    safe = json_safe(obj)

    assert isinstance(safe["path"], str)
    assert safe["path"] == "/tmp/test.txt"


def test_json_safe_sanitizes_nan():
    """Test json_safe converts NaN to None."""
    obj = {
        "value": float("nan"),
        "infinity": float("inf"),
        "normal": 3.14,
    }

    safe = json_safe(obj)

    assert safe["value"] is None
    assert safe["infinity"] is None
    assert safe["normal"] == 3.14


def test_operator_status_present():
    """Test manifests include operator_status with expected structure."""
    manifest = build_single_neuron_run(
        cell_type="izhikevich",
        params={},
        stimulus_pattern={},
        duration_ms=500.0,
        dt_ms=0.1,
    )

    assert "operator_status" in manifest
    op_status = manifest["operator_status"]

    # Check for Stage 2 operators
    for op_key in ("E_theta", "Q_eta_alpha", "F_field", "P_probe", "C_constraints"):
        assert op_key in op_status, f"Missing operator {op_key}"
        assert "status" in op_status[op_key]
        assert "claim_allowed" in op_status[op_key]
        assert "claim_forbidden" in op_status[op_key]
