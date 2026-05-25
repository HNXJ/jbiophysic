"""Manifest builders and jaxfne dispatch orchestration.

Critical doctrine:
- Builders construct manifests but do NOT execute.
- run_and_report() attempts dispatch but NEVER reports success=True without real jaxfne execution.
- Failed/unavailable dispatch returns success=False with explicit dispatch_status.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Optional
from uuid import uuid4

from . import config, convert
from .reports import get_installed_jaxfne_version, harmonize_jaxfne_output, write_manifest
from .validation import validate_manifest_json, validate_report

# Stage 2 operational version (incremented when bridge APIs change)
STAGE2_BRIDGE_CODE_VERSION = "0.1.0"


def _compute_config_hash(cfg: Dict[str, Any]) -> str:
    """Compute SHA256 hash of config dict for manifest.

    Parameters
    ----------
    cfg : Dict[str, Any]
        Config dict.

    Returns
    -------
    str
        Hex SHA256 hash.
    """
    cfg_json = json.dumps(cfg, sort_keys=True, default=str)
    return hashlib.sha256(cfg_json.encode()).hexdigest()


def _get_default_operator_status() -> Dict[str, Any]:
    """Get Stage 2 default operator status.

    E_theta: partial (bridges/jaxfne)
    Q_eta_alpha: future (specified but not implemented)
    F_field: future (specified but not implemented)
    P_probe: future (specified but not implemented)
    C_constraints: partial (bridges/jaxfne/validation.py)
    """
    return {
        "E_theta": {
            "status": "partial_repo_module",
            "implemented_paths": ["src/jbiophysic/bridges/jaxfne"],
            "claim_allowed": ["configuration_dispatch", "manifest_construction"],
            "claim_forbidden": ["biological_mechanism", "physical_amplitude"],
        },
        "Q_eta_alpha": {
            "status": "specified_future_module",
            "implemented_paths": [],
            "claim_allowed": [],
            "claim_forbidden": ["field_amplitude_claim"],
        },
        "F_field": {
            "status": "specified_future_module",
            "implemented_paths": [],
            "claim_allowed": [],
            "claim_forbidden": ["solved_field_claim"],
        },
        "P_probe": {
            "status": "specified_future_module",
            "implemented_paths": [],
            "claim_allowed": [],
            "claim_forbidden": ["empirical_lfp_csd_claim"],
        },
        "C_constraints": {
            "status": "partial_repo_module",
            "implemented_paths": ["src/jbiophysic/bridges/jaxfne/validation.py"],
            "claim_allowed": ["truth_gate_validation"],
            "claim_forbidden": ["biological_validation"],
        },
    }


def build_single_neuron_run(
    cell_type: str,
    params: Dict[str, Any],
    stimulus_pattern: Dict[str, Any],
    duration_ms: float,
    dt_ms: float,
    seed: int = 0,
) -> Dict[str, Any]:
    """Build schema-complete single-neuron manifest.

    Parameters
    ----------
    cell_type : str
        "izhikevich" or "hodgkin_huxley".
    params : Dict[str, Any]
        Cell parameters.
    stimulus_pattern : Dict[str, Any]
        Stimulus config (e.g. {"kind": "step", "start_ms": 100, "stop_ms": 400}).
    duration_ms : float
        Simulation duration (ms).
    dt_ms : float
        Integration timestep (ms).
    seed : int, optional
        Random seed. Default 0.

    Returns
    -------
    Dict[str, Any]
        Schema-complete manifest.

    Raises
    ------
    ValueError
        If parameters invalid.
    """
    # Validate config
    cfg = config.SingleNeuronConfig(
        cell_type=cell_type,
        params=params,
        stimulus_pattern=stimulus_pattern,
        duration_ms=duration_ms,
        dt_ms=dt_ms,
        seed=seed,
    )
    is_valid, errors = cfg.validate()
    if not is_valid:
        raise ValueError(f"Invalid single-neuron config: {errors}")

    # Compute metadata
    run_id = f"single_neuron_{cell_type}_{uuid4().hex[:8]}"
    config_dict = {
        "cell_type": cell_type,
        "params": params,
        "stimulus_pattern": stimulus_pattern,
        "duration_ms": duration_ms,
        "dt_ms": dt_ms,
        "seed": seed,
    }
    config_hash = _compute_config_hash(config_dict)
    n_steps = int(round(duration_ms / dt_ms))

    # Build jaxfne-compatible params
    jaxfne_params = convert.jbiophysic_params_to_jaxfne(cell_type, params)

    # Determine calibration status
    source_calibration_status = "toy_scale_not_empirical"  # Default for Stage 2
    source_decomposition = "total_membrane_current"
    field_solver_status = "not_solved"

    # Build manifest
    manifest = {
        "run_id": run_id,
        "run_type": "single_neuron",
        "source_type": cell_type,
        "source_scale": "toy",
        "jaxfne_engine_version": get_installed_jaxfne_version(),
        "jbiophysic_bridge_version": "bridges.jaxfne.v0.1",
        "truth_mode": "truth_safe_unverified",
        "claim_level": "computational_scaffold",
        "physical_amplitude_claim_allowed": False,
        "source_calibration_status": source_calibration_status,
        "source_decomposition": source_decomposition,
        "field_solver_status": field_solver_status,
        "seed": seed,
        "code_version": STAGE2_BRIDGE_CODE_VERSION,
        "config_hash": config_hash,
        "parameters": {
            "duration_ms": duration_ms,
            "dt_ms": dt_ms,
            "n_steps": n_steps,
            "cell_type": cell_type,
        },
        "jaxfne_request": {
            "cell_type": cell_type,
            "params": jaxfne_params,
            "stimulus": stimulus_pattern,
            "duration_ms": duration_ms,
            "dt_ms": dt_ms,
            "n_steps": n_steps,
        },
        "jaxfne_output": {},
        "harmonized_output": {},
        "operator_status": _get_default_operator_status(),
        "outputs": {},
    }

    return manifest


def build_ei_network_run(
    n_exc: int,
    n_inh: int,
    connectivity_config: Dict[str, Any],
    stimulus_config: Dict[str, Any],
    duration_ms: float,
    dt_ms: float,
    seed: int = 0,
) -> Dict[str, Any]:
    """Build schema-complete E/I network manifest.

    Parameters
    ----------
    n_exc : int
        Number of excitatory neurons.
    n_inh : int
        Number of inhibitory neurons.
    connectivity_config : Dict[str, Any]
        Connectivity dict (adjacency, weights, synapse_model).
    stimulus_config : Dict[str, Any]
        Stimulus configuration.
    duration_ms : float
        Simulation duration (ms).
    dt_ms : float
        Integration timestep (ms).
    seed : int, optional
        Random seed. Default 0.

    Returns
    -------
    Dict[str, Any]
        Schema-complete manifest.

    Raises
    ------
    ValueError
        If parameters invalid.
    """
    # Validate config
    cfg = config.EINetworkConfig(
        n_exc=n_exc,
        n_inh=n_inh,
        connectivity_config=connectivity_config,
        stimulus_config=stimulus_config,
        duration_ms=duration_ms,
        dt_ms=dt_ms,
        seed=seed,
    )
    is_valid, errors = cfg.validate()
    if not is_valid:
        raise ValueError(f"Invalid E/I network config: {errors}")

    # Compute metadata
    run_id = f"ei_network_{n_exc}e_{n_inh}i_{uuid4().hex[:8]}"
    config_dict = {
        "n_exc": n_exc,
        "n_inh": n_inh,
        "connectivity_config": connectivity_config,
        "stimulus_config": stimulus_config,
        "duration_ms": duration_ms,
        "dt_ms": dt_ms,
        "seed": seed,
    }
    config_hash = _compute_config_hash(config_dict)
    n_steps = int(round(duration_ms / dt_ms))

    # Build jaxfne-compatible circuit
    jaxfne_circuit = convert.jbiophysic_circuit_to_jaxfne(n_exc, n_inh, connectivity_config)

    # Determine calibration status
    source_calibration_status = "uncalibrated_spike_only"  # Default for Stage 2
    source_decomposition = "total_membrane_current"
    field_solver_status = "not_solved"

    # Build manifest
    manifest = {
        "run_id": run_id,
        "run_type": "ei_network",
        "source_type": "izhikevich_network",
        "source_scale": "proxy",
        "jaxfne_engine_version": get_installed_jaxfne_version(),
        "jbiophysic_bridge_version": "bridges.jaxfne.v0.1",
        "truth_mode": "truth_safe_unverified",
        "claim_level": "computational_scaffold",
        "physical_amplitude_claim_allowed": False,
        "source_calibration_status": source_calibration_status,
        "source_decomposition": source_decomposition,
        "field_solver_status": field_solver_status,
        "seed": seed,
        "code_version": STAGE2_BRIDGE_CODE_VERSION,
        "config_hash": config_hash,
        "parameters": {
            "duration_ms": duration_ms,
            "dt_ms": dt_ms,
            "n_steps": n_steps,
            "n_exc": n_exc,
            "n_inh": n_inh,
            "n_neurons": n_exc + n_inh,
        },
        "jaxfne_request": {
            "circuit": jaxfne_circuit,
            "stimulus": stimulus_config,
            "duration_ms": duration_ms,
            "dt_ms": dt_ms,
            "n_steps": n_steps,
        },
        "jaxfne_output": {},
        "harmonized_output": {},
        "operator_status": _get_default_operator_status(),
        "outputs": {},
    }

    return manifest


def build_laminar_proxy_run(
    laminar_config: Dict[str, Any],
    source_scale: str,
    stimulus_pattern: Dict[str, Any],
    duration_ms: float,
    dt_ms: float,
    seed: int = 0,
) -> Dict[str, Any]:
    """Build schema-complete laminar proxy manifest.

    Parameters
    ----------
    laminar_config : Dict[str, Any]
        Laminar circuit configuration.
    source_scale : str
        One of "toy", "proxy", "calibrated", "physical".
    stimulus_pattern : Dict[str, Any]
        Stimulus configuration.
    duration_ms : float
        Simulation duration (ms).
    dt_ms : float
        Integration timestep (ms).
    seed : int, optional
        Random seed. Default 0.

    Returns
    -------
    Dict[str, Any]
        Schema-complete manifest.

    Raises
    ------
    ValueError
        If parameters invalid.
    """
    # Validate config
    cfg = config.LaminarProxyConfig(
        source_scale=source_scale,
        laminar_config=laminar_config,
        stimulus_pattern=stimulus_pattern,
        duration_ms=duration_ms,
        dt_ms=dt_ms,
        seed=seed,
    )
    is_valid, errors = cfg.validate()
    if not is_valid:
        raise ValueError(f"Invalid laminar proxy config: {errors}")

    # Compute metadata
    run_id = f"laminar_{source_scale}_{uuid4().hex[:8]}"
    config_dict = {
        "laminar_config": laminar_config,
        "source_scale": source_scale,
        "stimulus_pattern": stimulus_pattern,
        "duration_ms": duration_ms,
        "dt_ms": dt_ms,
        "seed": seed,
    }
    config_hash = _compute_config_hash(config_dict)
    n_steps = int(round(duration_ms / dt_ms))

    # Map source_scale to calibration status
    scale_to_calibration = {
        "toy": "toy_scale_not_empirical",
        "proxy": "uncalibrated_spike_only",
        "calibrated": "empirically_calibrated",
        "physical": "physical",
    }
    source_calibration_status = scale_to_calibration.get(source_scale, "uncalibrated_spike_only")
    source_decomposition = "proxy_no_field_solve"
    field_solver_status = "laminar_proxy_no_pde"

    # Build manifest
    manifest = {
        "run_id": run_id,
        "run_type": "laminar_proxy",
        "source_type": "proxy_source",
        "source_scale": source_scale,
        "jaxfne_engine_version": get_installed_jaxfne_version(),
        "jbiophysic_bridge_version": "bridges.jaxfne.v0.1",
        "truth_mode": "truth_safe_unverified",
        "claim_level": "computational_scaffold",
        "physical_amplitude_claim_allowed": False,
        "source_calibration_status": source_calibration_status,
        "source_decomposition": source_decomposition,
        "field_solver_status": field_solver_status,
        "seed": seed,
        "code_version": STAGE2_BRIDGE_CODE_VERSION,
        "config_hash": config_hash,
        "parameters": {
            "duration_ms": duration_ms,
            "dt_ms": dt_ms,
            "n_steps": n_steps,
            "source_scale": source_scale,
        },
        "jaxfne_request": {
            "laminar_config": laminar_config,
            "stimulus": stimulus_pattern,
            "duration_ms": duration_ms,
            "dt_ms": dt_ms,
            "n_steps": n_steps,
        },
        "jaxfne_output": {},
        "harmonized_output": {},
        "operator_status": _get_default_operator_status(),
        "outputs": {},
    }

    return manifest


def get_jaxfne_report(
    manifest_dict: Dict[str, Any],
    cache_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """Fetch or compute jaxfne report from manifest.

    Stage 2: Attempts jaxfne import and checks for supported execution API.

    Parameters
    ----------
    manifest_dict : Dict[str, Any]
        Manifest dict with jaxfne_request section.
    cache_dir : str, optional
        Cache directory (unused in Stage 2). Default None.

    Returns
    -------
    Dict[str, Any]
        Report dict with dispatch_status and success=False unless real jaxfne executed.
    """
    # Attempt jaxfne import
    try:
        import jaxfne as jtfne

        jaxfne_available = True
        jaxfne_version = get_installed_jaxfne_version()
    except ImportError as e:
        jaxfne_available = False
        jaxfne_version = "unknown"
        return {
            "dispatch_status": "jaxfne_unavailable",
            "success": False,
            "errors": [f"jaxfne import failed: {e}"],
            "jaxfne_version": jaxfne_version,
            "truth_mode": "truth_safe_unverified",
            "claim_level": "computational_scaffold",
        }

    # Check for supported execution API
    # Stage 2: jaxfne may not expose execution APIs yet
    supported_apis = []
    if hasattr(jtfne, "run_single_neuron"):
        supported_apis.append("run_single_neuron")
    if hasattr(jtfne, "run_ei_network"):
        supported_apis.append("run_ei_network")
    if hasattr(jtfne, "run_laminar"):
        supported_apis.append("run_laminar")

    if not supported_apis:
        return {
            "dispatch_status": "no_supported_jaxfne_execution_api",
            "success": False,
            "errors": [
                "jaxfne imported but no supported execution API found. "
                "Available APIs in jaxfne:", list(dir(jtfne))
            ],
            "jaxfne_version": jaxfne_version,
            "truth_mode": "truth_safe_unverified",
            "claim_level": "computational_scaffold",
        }

    # If we reach here, real jaxfne APIs exist and could execute
    # For Stage 2, we don't actually call them yet, but we document that they exist
    return {
        "dispatch_status": "jaxfne_available_but_stage2_no_execution",
        "success": False,
        "supported_apis": supported_apis,
        "jaxfne_version": jaxfne_version,
        "truth_mode": "truth_safe_unverified",
        "claim_level": "computational_scaffold",
        "notes": "jaxfne execution APIs found but Stage 2 defers actual dispatch to later stages",
    }


def run_and_report(
    manifest: Dict[str, Any],
    output_dir: str,
) -> Dict[str, Any]:
    """Execute jaxfne dispatch and return harmonized report.

    CRITICAL: Does NOT report success=True unless real jaxfne API executed.

    Parameters
    ----------
    manifest : Dict[str, Any]
        Manifest to execute.
    output_dir : str
        Directory for output artifacts.

    Returns
    -------
    Dict[str, Any]
        Report with dispatch_status, success, truth metadata, and written artifacts.
    """
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Validate manifest before running
    is_valid, errors = validate_manifest_json(manifest, strict_mode=True)
    if not is_valid:
        report = {
            "dispatch_status": "manifest_invalid",
            "success": False,
            "errors": errors,
            "truth_mode": "truth_safe_unverified",
            "claim_level": "computational_scaffold",
        }
        write_manifest(report, str(output_path / "jaxfne_bridge_report.json"), allow_nan=False)
        return report

    # Attempt jaxfne dispatch
    jaxfne_report = get_jaxfne_report(manifest)

    # Build final report
    report = {
        "dispatch_status": jaxfne_report.get("dispatch_status"),
        "success": False,  # Stage 2: never report success
        "truth_mode": "truth_safe_unverified",
        "claim_level": "computational_scaffold",
        "manifest": manifest,
        "jaxfne_dispatch": jaxfne_report,
    }

    # Validate full report (will fail if success=True with bad dispatch status)
    is_valid, report_errors = validate_report(report, strict_mode=True)
    if not is_valid:
        report["validation_errors"] = report_errors
        report["success"] = False  # Ensure false

    # Write artifacts
    write_manifest(manifest, str(output_path / "manifest.json"), allow_nan=False)
    write_manifest(report, str(output_path / "jaxfne_bridge_report.json"), allow_nan=False)

    return report
