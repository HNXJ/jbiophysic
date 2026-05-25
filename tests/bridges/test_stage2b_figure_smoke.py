"""Automated Stage 2B bridge API tests.

These tests verify that the bridge API generates manifests with correct
structure, validation, and honest dispatch classification.
"""

import json
import tempfile
from pathlib import Path

import pytest

from jbiophysic.bridges.jaxfne import (
    build_single_neuron_run,
    run_and_report,
    validate_manifest_json,
    validate_report,
)


def test_stage2b_bridge_api_builds_valid_manifest():
    """Test that build_single_neuron_run creates valid manifest."""
    manifest = build_single_neuron_run(
        cell_type="izhikevich",
        params={
            "a": 0.02,
            "b": 0.2,
            "c": -65.0,
            "d": 8.0,
            "I_inj_nA": 10.0,
        },
        stimulus_pattern={
            "kind": "step",
            "start_ms": 100.0,
            "stop_ms": 400.0,
        },
        duration_ms=500.0,
        dt_ms=0.1,
        seed=0,
    )

    is_valid, errors = validate_manifest_json(manifest, strict_mode=True)
    assert is_valid, f"Manifest validation failed: {errors}"
    assert manifest["truth_mode"] == "truth_safe_unverified"
    assert manifest["claim_level"] == "computational_scaffold"


def test_stage2b_run_and_report_writes_manifest_json():
    """Test that run_and_report writes manifest.json."""
    with tempfile.TemporaryDirectory() as tmpdir:
        manifest = build_single_neuron_run(
            cell_type="izhikevich",
            params={},
            stimulus_pattern={},
            duration_ms=500.0,
            dt_ms=0.1,
        )

        report = run_and_report(manifest, tmpdir)

        # Manifest file should exist
        manifest_path = Path(tmpdir) / "manifest.json"
        assert manifest_path.exists(), "manifest.json not written by run_and_report"

        # Should be valid JSON
        with open(manifest_path) as f:
            loaded = json.load(f)

        assert isinstance(loaded, dict)
        assert loaded["truth_mode"] == "truth_safe_unverified"


def test_stage2b_run_and_report_writes_report_json():
    """Test that run_and_report writes jaxfne_bridge_report.json."""
    with tempfile.TemporaryDirectory() as tmpdir:
        manifest = build_single_neuron_run(
            cell_type="izhikevich",
            params={},
            stimulus_pattern={},
            duration_ms=500.0,
            dt_ms=0.1,
        )

        report = run_and_report(manifest, tmpdir)

        # Report file should exist
        report_path = Path(tmpdir) / "jaxfne_bridge_report.json"
        assert report_path.exists(), "jaxfne_bridge_report.json not written by run_and_report"

        # Should be valid JSON
        with open(report_path) as f:
            loaded = json.load(f)

        assert isinstance(loaded, dict)
        assert "dispatch_status" in loaded
        assert "success" in loaded


def test_stage2b_manifest_has_operator_status():
    """Test that built manifest includes operator_status."""
    manifest = build_single_neuron_run(
        cell_type="izhikevich",
        params={},
        stimulus_pattern={},
        duration_ms=500.0,
        dt_ms=0.1,
    )

    assert "operator_status" in manifest
    op_status = manifest["operator_status"]
    assert isinstance(op_status, dict)
    assert len(op_status) > 0


def test_stage2b_no_simulation_when_no_jaxfne_api():
    """CRITICAL: No jaxfne API means no simulation_executed.

    Verify that when dispatch_status indicates no supported jaxfne API,
    success is False and report reflects unavailability.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        manifest = build_single_neuron_run(
            cell_type="izhikevich",
            params={},
            stimulus_pattern={},
            duration_ms=500.0,
            dt_ms=0.1,
        )

        report = run_and_report(manifest, tmpdir)

        # Check dispatch status
        dispatch_status = report.get("dispatch_status")
        success = report.get("success")

        # If no supported API, success must be False
        if dispatch_status in (
            "jaxfne_unavailable",
            "no_supported_jaxfne_execution_api",
        ):
            assert (
                success is False
            ), f"success must be False when dispatch_status={dispatch_status}, got {success}"


def test_stage2b_no_fake_success_on_dispatch_failure():
    """CRITICAL: Reject fake success when dispatch fails.

    run_and_report must never report success=True when jaxfne API is unavailable.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        manifest = build_single_neuron_run(
            cell_type="izhikevich",
            params={},
            stimulus_pattern={},
            duration_ms=500.0,
            dt_ms=0.1,
        )

        report = run_and_report(manifest, tmpdir)

        # Load written report for final check
        report_path = Path(tmpdir) / "jaxfne_bridge_report.json"
        with open(report_path) as f:
            written_report = json.load(f)

        # The critical gate: fake success detection
        dispatch_status = written_report.get("dispatch_status")
        success = written_report.get("success")

        # These combinations are forbidden (checked in validation)
        forbidden_combos = [
            ("jaxfne_unavailable", True),
            ("no_supported_jaxfne_execution_api", True),
            ("not_executed_stub", True),
        ]

        for bad_status, bad_success in forbidden_combos:
            if dispatch_status == bad_status and success == bad_success:
                pytest.fail(
                    f"FATAL: success={success} with dispatch_status={dispatch_status} detected. "
                    f"This is the fake-success condition Stage 2 forbids."
                )


def test_stage2b_manifest_strict_json_roundtrip():
    """Test that manifest JSON can round-trip without NaN/Inf."""
    with tempfile.TemporaryDirectory() as tmpdir:
        manifest = build_single_neuron_run(
            cell_type="izhikevich",
            params={},
            stimulus_pattern={},
            duration_ms=500.0,
            dt_ms=0.1,
        )

        run_and_report(manifest, tmpdir)

        # Load manifest and verify it round-trips
        with open(Path(tmpdir) / "manifest.json") as f:
            loaded = json.load(f)

        # Re-serialize and verify
        reserialized = json.dumps(loaded)
        reloaded = json.loads(reserialized)

        assert loaded == reloaded, "JSON round-trip failed"
