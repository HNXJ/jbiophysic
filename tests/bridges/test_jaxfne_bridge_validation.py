"""Tests for jaxfne bridge validation.

CRITICAL: Includes test_reject_fake_success_with_no_jaxfne_api() that guards against
the exact failure mode Stage 2 forbids: success=True with failed/unavailable dispatch.
"""

from jbiophysic.bridges.jaxfne import (
    build_single_neuron_run,
    validate_manifest_json,
    validate_report,
)


def test_validate_manifest_json_passes_valid():
    """Test validate_manifest_json accepts valid manifests."""
    manifest = build_single_neuron_run(
        cell_type="izhikevich",
        params={},
        stimulus_pattern={},
        duration_ms=500.0,
        dt_ms=0.1,
    )

    is_valid, errors = validate_manifest_json(manifest)
    assert is_valid, f"Valid manifest should pass: {errors}"
    assert len(errors) == 0


def test_validate_manifest_rejects_wrong_truth_mode():
    """Test validate_manifest rejects wrong truth_mode."""
    manifest = build_single_neuron_run(
        cell_type="izhikevich",
        params={},
        stimulus_pattern={},
        duration_ms=500.0,
        dt_ms=0.1,
    )
    manifest["truth_mode"] = "wrong_mode"

    is_valid, errors = validate_manifest_json(manifest)
    assert not is_valid
    assert any("truth_mode" in e for e in errors)


def test_validate_manifest_rejects_wrong_claim_level():
    """Test validate_manifest rejects wrong claim_level."""
    manifest = build_single_neuron_run(
        cell_type="izhikevich",
        params={},
        stimulus_pattern={},
        duration_ms=500.0,
        dt_ms=0.1,
    )
    manifest["claim_level"] = "verified_biological"

    is_valid, errors = validate_manifest_json(manifest)
    assert not is_valid
    assert any("claim_level" in e for e in errors)


def test_validate_manifest_rejects_amplitude_claim():
    """Test validate_manifest rejects physical_amplitude_claim_allowed=True."""
    manifest = build_single_neuron_run(
        cell_type="izhikevich",
        params={},
        stimulus_pattern={},
        duration_ms=500.0,
        dt_ms=0.1,
    )
    manifest["physical_amplitude_claim_allowed"] = True

    is_valid, errors = validate_manifest_json(manifest)
    assert not is_valid
    assert any("physical_amplitude_claim_allowed" in e for e in errors)


def test_validate_manifest_rejects_invalid_run_type():
    """Test validate_manifest rejects invalid run_type."""
    manifest = build_single_neuron_run(
        cell_type="izhikevich",
        params={},
        stimulus_pattern={},
        duration_ms=500.0,
        dt_ms=0.1,
    )
    manifest["run_type"] = "unknown_run_type"

    is_valid, errors = validate_manifest_json(manifest)
    assert not is_valid
    assert any("run_type" in e for e in errors)


def test_validate_manifest_rejects_invalid_source_scale():
    """Test validate_manifest rejects invalid source_scale."""
    manifest = build_single_neuron_run(
        cell_type="izhikevich",
        params={},
        stimulus_pattern={},
        duration_ms=500.0,
        dt_ms=0.1,
    )
    manifest["source_scale"] = "invalid_scale"

    is_valid, errors = validate_manifest_json(manifest)
    assert not is_valid
    assert any("source_scale" in e for e in errors)


def test_validate_manifest_rejects_missing_fields():
    """Test validate_manifest rejects manifests with missing fields."""
    manifest = build_single_neuron_run(
        cell_type="izhikevich",
        params={},
        stimulus_pattern={},
        duration_ms=500.0,
        dt_ms=0.1,
    )
    # Remove a required field
    del manifest["config_hash"]

    is_valid, errors = validate_manifest_json(manifest)
    assert not is_valid
    assert any("Missing required fields" in e for e in errors)


def test_validate_report_passes_valid():
    """Test validate_report accepts valid reports."""
    manifest = build_single_neuron_run(
        cell_type="izhikevich",
        params={},
        stimulus_pattern={},
        duration_ms=500.0,
        dt_ms=0.1,
    )

    report = {
        "success": False,
        "dispatch_status": "not_executed_stub",
        "truth_mode": "truth_safe_unverified",
        "claim_level": "computational_scaffold",
        "manifest": manifest,
    }

    is_valid, errors = validate_report(report)
    assert is_valid, f"Valid report should pass: {errors}"


def test_validate_report_rejects_wrong_truth_mode():
    """Test validate_report rejects wrong truth_mode."""
    report = {
        "success": False,
        "dispatch_status": "not_executed_stub",
        "truth_mode": "wrong_mode",
        "claim_level": "computational_scaffold",
    }

    is_valid, errors = validate_report(report)
    assert not is_valid
    assert any("truth_mode" in e for e in errors)


def test_validate_report_rejects_wrong_claim_level():
    """Test validate_report rejects wrong claim_level."""
    report = {
        "success": False,
        "dispatch_status": "not_executed_stub",
        "truth_mode": "truth_safe_unverified",
        "claim_level": "verified_biological",
    }

    is_valid, errors = validate_report(report)
    assert not is_valid
    assert any("claim_level" in e for e in errors)


def test_validate_report_rejects_amplitude_claim():
    """Test validate_report rejects physical_amplitude_claim_allowed=True."""
    report = {
        "success": False,
        "dispatch_status": "not_executed_stub",
        "truth_mode": "truth_safe_unverified",
        "claim_level": "computational_scaffold",
        "physical_amplitude_claim_allowed": True,
    }

    is_valid, errors = validate_report(report, strict_mode=True)
    assert not is_valid
    assert any("physical_amplitude_claim_allowed" in e for e in errors)


def test_validate_report_rejects_fake_success_with_jaxfne_unavailable():
    """CRITICAL: Reject success=True when dispatch_status=jaxfne_unavailable."""
    fake_report = {
        "success": True,
        "dispatch_status": "jaxfne_unavailable",
        "truth_mode": "truth_safe_unverified",
        "claim_level": "computational_scaffold",
    }

    is_valid, errors = validate_report(fake_report, strict_mode=True)
    assert not is_valid, f"Should reject success=True with jaxfne_unavailable. Errors: {errors}"
    assert any("success" in e.lower() and "dispatch" in e.lower() for e in errors), (
        f"Error message must mention success/dispatch contradiction. Got: {errors}"
    )


def test_validate_report_rejects_fake_success_with_no_jaxfne_api():
    """CRITICAL: Reject success=True when dispatch_status=no_supported_jaxfne_execution_api.

    This is the exact failure mode Stage 2 forbids: a stub reporting fake success
    while dispatch_status indicates jaxfne is unavailable or has no supported API.
    """
    fake_report = {
        "success": True,
        "dispatch_status": "no_supported_jaxfne_execution_api",
        "truth_mode": "truth_safe_unverified",
        "claim_level": "computational_scaffold",
    }

    is_valid, errors = validate_report(fake_report, strict_mode=True)
    assert not is_valid, f"Should reject success=True with no-API status. Errors: {errors}"
    assert any("success" in e.lower() and "dispatch" in e.lower() for e in errors), (
        f"Error message must mention success/dispatch contradiction. Got: {errors}"
    )
    # Also check for the FATAL marker
    assert any("FATAL" in e for e in errors), f"Error should be marked FATAL. Got: {errors}"


def test_validate_report_rejects_fake_success_with_not_executed_stub():
    """CRITICAL: Reject success=True when dispatch_status=not_executed_stub."""
    fake_report = {
        "success": True,
        "dispatch_status": "not_executed_stub",
        "truth_mode": "truth_safe_unverified",
        "claim_level": "computational_scaffold",
    }

    is_valid, errors = validate_report(fake_report, strict_mode=True)
    assert not is_valid, f"Should reject success=True with not_executed_stub. Errors: {errors}"
    assert any("success" in e.lower() and "dispatch" in e.lower() for e in errors), (
        f"Error message must mention success/dispatch contradiction. Got: {errors}"
    )


def test_validate_report_allows_success_false_with_any_dispatch_status():
    """Test validate_report allows success=False regardless of dispatch_status."""
    for dispatch_status in (
        "jaxfne_unavailable",
        "no_supported_jaxfne_execution_api",
        "not_executed_stub",
        "manifest_invalid",
    ):
        report = {
            "success": False,
            "dispatch_status": dispatch_status,
            "truth_mode": "truth_safe_unverified",
            "claim_level": "computational_scaffold",
        }

        is_valid, errors = validate_report(report)
        assert is_valid, f"success=False with {dispatch_status} should be valid: {errors}"


def test_validate_report_rejects_unknown_dispatch_status():
    """Test validate_report rejects unknown dispatch_status."""
    report = {
        "success": False,
        "dispatch_status": "unknown",
        "truth_mode": "truth_safe_unverified",
        "claim_level": "computational_scaffold",
    }

    is_valid, errors = validate_report(report)
    assert not is_valid
    assert any("dispatch_status" in e for e in errors)
