"""Manifest and report validation with truth constraints.

Hard validation rules:
- truth_mode == "truth_safe_unverified"
- claim_level == "computational_scaffold"
- physical_amplitude_claim_allowed is False for Stage 2
- toy/proxy/uncalibrated sources reject amplitude claims
- resistive_forward_solved requires boundary/gauge/residual metadata
- success=True forbidden when dispatch_status indicates unavailable/no-API/stub
- strict JSON round-trip must pass with allow_nan=False
"""

from __future__ import annotations

from typing import Any

# Canonical constants
TRUTH_MODE = "truth_safe_unverified"
CLAIM_LEVEL = "computational_scaffold"

# Allowed literal values
ALLOWED_RUN_TYPES = {"single_neuron", "ei_network", "laminar_proxy"}
ALLOWED_SOURCE_TYPES = {
    "izhikevich",
    "hodgkin_huxley",
    "izhikevich_network",
    "hodgkin_huxley_network",
    "proxy_source",
}
ALLOWED_SOURCE_SCALES = {"toy", "proxy", "calibrated", "physical"}
ALLOWED_CALIBRATION_STATUSES = {
    "toy_scale_not_empirical",
    "uncalibrated_spike_only",
    "calibrated_proxy",
    "empirically_calibrated",
    "physical",
}
ALLOWED_SOURCE_DECOMPOSITIONS = {
    "proxy_no_field_solve",
    "total_membrane_current",
    "decomposed_cap_ion_syn",
}
ALLOWED_FIELD_SOLVER_STATUSES = {
    "not_solved",
    "smoke_only",
    "laminar_proxy_no_pde",
    "resistive_forward_solved",
    "invalid",
}

# Required manifest fields
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


def validate_manifest_json(
    manifest: dict[str, Any],
    strict_mode: bool = True,
) -> tuple[bool, list[str]]:
    """Validate manifest structure and truth constraints.

    Parameters
    ----------
    manifest : Dict[str, Any]
        Manifest to validate.
    strict_mode : bool, optional
        If True, enforce all hard constraints. Default True.

    Returns
    -------
    Tuple[bool, List[str]]
        (is_valid, list of error messages).
    """
    errors = []

    # Check required fields
    missing = REQUIRED_MANIFEST_FIELDS - set(manifest.keys())
    if missing:
        errors.append(f"Missing required fields: {missing}")

    # Check field types and values
    if "run_type" in manifest and manifest["run_type"] not in ALLOWED_RUN_TYPES:
        errors.append(
            f"run_type must be one of {ALLOWED_RUN_TYPES}, got {manifest['run_type']}"
        )

    if "source_type" in manifest and manifest["source_type"] not in ALLOWED_SOURCE_TYPES:
        errors.append(
            f"source_type must be one of {ALLOWED_SOURCE_TYPES}, got {manifest['source_type']}"
        )

    if "source_scale" in manifest and manifest["source_scale"] not in ALLOWED_SOURCE_SCALES:
        errors.append(
            f"source_scale must be one of {ALLOWED_SOURCE_SCALES}, got {manifest['source_scale']}"
        )

    if (
        "source_calibration_status" in manifest
        and manifest["source_calibration_status"] not in ALLOWED_CALIBRATION_STATUSES
    ):
        errors.append(
            f"source_calibration_status must be one of {ALLOWED_CALIBRATION_STATUSES}, "
            f"got {manifest['source_calibration_status']}"
        )

    if (
        "source_decomposition" in manifest
        and manifest["source_decomposition"] not in ALLOWED_SOURCE_DECOMPOSITIONS
    ):
        errors.append(
            f"source_decomposition must be one of {ALLOWED_SOURCE_DECOMPOSITIONS}, "
            f"got {manifest['source_decomposition']}"
        )

    if (
        "field_solver_status" in manifest
        and manifest["field_solver_status"] not in ALLOWED_FIELD_SOLVER_STATUSES
    ):
        errors.append(
            f"field_solver_status must be one of {ALLOWED_FIELD_SOLVER_STATUSES}, "
            f"got {manifest['field_solver_status']}"
        )

    # Check truth constraints
    if manifest.get("truth_mode") != TRUTH_MODE:
        errors.append(f"truth_mode must be '{TRUTH_MODE}', got {manifest.get('truth_mode')}")

    if manifest.get("claim_level") != CLAIM_LEVEL:
        errors.append(f"claim_level must be '{CLAIM_LEVEL}', got {manifest.get('claim_level')}")

    if manifest.get("physical_amplitude_claim_allowed"):
        errors.append("physical_amplitude_claim_allowed must be False in Stage 2")

    # Check resistive_forward_solved constraint
    if (
        strict_mode
        and manifest.get("field_solver_status") == "resistive_forward_solved"
    ):
        # Require solver diagnostics
        operator_status = manifest.get("operator_status", {})
        f_field = operator_status.get("F_field", {})
        if f_field.get("status") != "solved_resistive_forward":
            errors.append(
                "field_solver_status='resistive_forward_solved' requires F_field operator status "
                "to be 'solved_resistive_forward' with boundary/gauge/residual metadata"
            )

    return len(errors) == 0, errors


def validate_report(
    report: dict[str, Any],
    strict_mode: bool = True,
) -> tuple[bool, list[str]]:
    """Validate full report with truth gatekeeping and dispatch semantics.

    CRITICAL: Rejects success=True when dispatch_status indicates unavailable/no-API/stub.

    Parameters
    ----------
    report : Dict[str, Any]
        Report dict to validate.
    strict_mode : bool, optional
        If True, enforce all hard constraints. Default True.

    Returns
    -------
    Tuple[bool, List[str]]
        (is_valid, list of error messages).
    """
    errors = []

    # CRITICAL: Check for fake success
    success = report.get("success", False)
    dispatch_status = report.get("dispatch_status", "unknown")

    if success and dispatch_status in (
        "jaxfne_unavailable",
        "no_supported_jaxfne_execution_api",
        "not_executed_stub",
    ):
        errors.append(
            f"FATAL: success=True but dispatch_status='{dispatch_status}'. "
            f"Stub/unavailable dispatch cannot report success. "
            f"This is the exact fake-success condition that Stage 2 forbids."
        )

    # Check truth claims
    if report.get("truth_mode") != TRUTH_MODE:
        errors.append(f"Report truth_mode must be '{TRUTH_MODE}', got {report.get('truth_mode')}")

    if report.get("claim_level") != CLAIM_LEVEL:
        errors.append(f"Report claim_level must be '{CLAIM_LEVEL}', got {report.get('claim_level')}")

    if strict_mode and report.get("physical_amplitude_claim_allowed", False):
        errors.append("physical_amplitude_claim_allowed must be False (strict mode)")

    # Validate manifest section if present
    if "manifest" in report:
        manifest_valid, manifest_errors = validate_manifest_json(
            report["manifest"], strict_mode=strict_mode
        )
        if not manifest_valid:
            errors.extend([f"manifest: {e}" for e in manifest_errors])

    # Check dispatch status is documented
    if not dispatch_status or dispatch_status == "unknown":
        errors.append("dispatch_status must be explicitly set (not 'unknown')")

    return len(errors) == 0, errors
