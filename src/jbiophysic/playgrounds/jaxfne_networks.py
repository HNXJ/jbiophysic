"""Thin jaxfne neural-circuit playground wrappers.

This module turns jbiophysic into a safe playground for jaxfne-backed neural
circuit models while keeping jaxfne as the engine. It does not reimplement
jaxfne emitters, source projection, fields, probes, objectives, or optimizers.

Canonical engine import inside runtime functions only:

    import jaxfne as jtfne

Status gates are deliberately conservative:
`truth_safe_unverified`, `computational_scaffold`, `laminar_proxy_no_pde`, and
`physical_amplitude_claim_allowed=False`.
"""

from __future__ import annotations

import importlib
import json
import platform
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np

from jbiophysic.io.manifests import json_safe

TRUTH_MODE = "truth_safe_unverified"
CLAIM_LEVEL = "computational_scaffold"
FIELD_STATUS = "laminar_proxy_no_pde"
PHYSICAL_AMPLITUDE_CLAIM_ALLOWED = False
SOURCE_CALIBRATION_STATUS = "uncalibrated_spike_only"

PlaygroundName = Literal[
    "suite2_single_neuron",
    "suite2_four_celltype",
    "suite2_net1",
    "suite2_v1_v4",
]

_SUPPORTED_PLAYGROUNDS: tuple[PlaygroundName, ...] = (
    "suite2_single_neuron",
    "suite2_four_celltype",
    "suite2_net1",
    "suite2_v1_v4",
)

_CONFIG_BUILDERS: dict[str, str] = {
    "suite2_single_neuron": "suite2_single_neuron_config",
    "suite2_four_celltype": "suite2_four_celltype_config",
    "suite2_net1": "suite2_net1_config",
    "suite2_v1_v4": "suite2_v1_v4_config",
}


@dataclass(frozen=True)
class JaxfnePlaygroundSpec:
    """Declarative request for a jaxfne-backed circuit playground run.

    The spec is JSON-safe by construction and is usable even when jaxfne is not
    installed. Runtime execution imports jaxfne lazily through :func:`require_jaxfne`.
    """

    name: PlaygroundName = "suite2_four_celltype"
    seed: int = 0
    duration_ms: float = 10.0
    dt_ms: float = 0.1
    cell_type: str = "E"
    noise_amplitude: float | None = None
    truth_mode: str = TRUTH_MODE
    claim_level: str = CLAIM_LEVEL
    field_status: str = FIELD_STATUS
    source_calibration_status: str = SOURCE_CALIBRATION_STATUS
    physical_amplitude_claim_allowed: bool = PHYSICAL_AMPLITUDE_CLAIM_ALLOWED
    metadata: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        """Raise a precise error for invalid request specs."""

        if self.name not in _SUPPORTED_PLAYGROUNDS:
            raise ValueError(
                f"unsupported jaxfne playground {self.name!r}; "
                f"expected one of {list(_SUPPORTED_PLAYGROUNDS)!r}"
            )
        if self.duration_ms <= 0:
            raise ValueError("duration_ms must be positive")
        if self.dt_ms <= 0:
            raise ValueError("dt_ms must be positive")
        if self.seed < 0:
            raise ValueError("seed must be non-negative")
        if self.truth_mode != TRUTH_MODE:
            raise ValueError(f"truth_mode must be {TRUTH_MODE!r}")
        if self.claim_level != CLAIM_LEVEL:
            raise ValueError(f"claim_level must be {CLAIM_LEVEL!r}")
        if self.field_status != FIELD_STATUS:
            raise ValueError(f"field_status must be {FIELD_STATUS!r}")
        if self.physical_amplitude_claim_allowed:
            raise ValueError("physical_amplitude_claim_allowed must remain False")

    def to_json_dict(self) -> dict[str, Any]:
        """Return a strict JSON-safe spec dictionary."""

        self.validate()
        return json_safe(asdict(self))


def available_playgrounds() -> tuple[str, ...]:
    """Return stable playground names supported by this adapter."""

    return tuple(_SUPPORTED_PLAYGROUNDS)


def require_jaxfne() -> Any:
    """Import jaxfne lazily and return it as ``jtfne``.

    The import stays inside this function so core jbiophysic remains importable
    without jaxfne/JAX extras. The local variable name should stay ``jtfne`` in
    runtime wrappers to match the project-level canonical import rule.
    """

    try:
        import jaxfne as jtfne  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "jaxfne is required for executable jaxfne playground runs. "
            "Install jbiophysic with the jaxfne extra, for example: "
            "pip install -e '.[jax,jaxfne]'"
        ) from exc
    return jtfne


def _get_jax_report() -> dict[str, Any]:
    try:
        jax = importlib.import_module("jax")
        jaxlib = importlib.import_module("jaxlib")
    except Exception as exc:  # pragma: no cover - environment dependent
        return {"available": False, "error": repr(exc)}

    try:
        devices = [str(device) for device in jax.devices()]
    except Exception as exc:  # pragma: no cover - environment dependent
        devices = [f"device_query_failed: {exc!r}"]

    return {
        "available": True,
        "jax_version": getattr(jax, "__version__", "unknown"),
        "jaxlib_version": getattr(jaxlib, "__version__", "unknown"),
        "devices": devices,
        "x64_enabled": bool(getattr(jax.config, "jax_enable_x64", False)),
    }


def jaxfne_backend_report(jaxfne_module: Any | None = None) -> dict[str, Any]:
    """Return dependency/API status without mutating global state."""

    if jaxfne_module is None:
        try:
            jaxfne_module = require_jaxfne()
            available = True
            error = None
        except ImportError as exc:
            jaxfne_module = None
            available = False
            error = str(exc)
    else:
        available = True
        error = None

    api_checks = {}
    if jaxfne_module is not None:
        for api_name in sorted(set(_CONFIG_BUILDERS.values()) | {"construct", "simulate"}):
            api_checks[api_name] = hasattr(jaxfne_module, api_name)

    return json_safe(
        {
            "available": available,
            "error": error,
            "jaxfne_version": getattr(jaxfne_module, "__version__", "unknown")
            if jaxfne_module is not None
            else "unavailable",
            "required_apis": api_checks,
            "jax": _get_jax_report(),
            "truth_mode": TRUTH_MODE,
            "claim_level": CLAIM_LEVEL,
            "field_status": FIELD_STATUS,
            "physical_amplitude_claim_allowed": PHYSICAL_AMPLITUDE_CLAIM_ALLOWED,
        }
    )


def build_request_manifest(spec: JaxfnePlaygroundSpec) -> dict[str, Any]:
    """Build a dependency-free manifest for a requested playground run."""

    spec.validate()
    return json_safe(
        {
            "kind": "jbiophysic_jaxfne_playground_request",
            "spec": spec.to_json_dict(),
            "engine": "jaxfne",
            "canonical_import": "import jaxfne as jtfne",
            "pipeline": ["Emitter", "Source", "Field", "Probe", "Objective", "Optimizer"],
            "engine_api": {
                "config_builder": _CONFIG_BUILDERS[spec.name],
                "construct": "construct",
                "simulate": "simulate",
            },
            "status_gates": {
                "truth_mode": TRUTH_MODE,
                "claim_level": CLAIM_LEVEL,
                "field_status": FIELD_STATUS,
                "source_calibration_status": spec.source_calibration_status,
                "physical_amplitude_claim_allowed": PHYSICAL_AMPLITUDE_CLAIM_ALLOWED,
            },
            "limits": {
                "no_real_eeg_meg_claim": True,
                "no_calibrated_amplitude_claim": True,
                "no_mechanism_proof_claim": True,
                "no_pde_solve_claim": True,
            },
        }
    )


def build_jaxfne_config(
    spec: JaxfnePlaygroundSpec,
    *,
    jaxfne_module: Any | None = None,
) -> Any:
    """Build the corresponding jaxfne configuration through jaxfne public APIs."""

    spec.validate()
    jtfne = jaxfne_module if jaxfne_module is not None else require_jaxfne()
    builder_name = _CONFIG_BUILDERS[spec.name]
    if not hasattr(jtfne, builder_name):
        raise AttributeError(f"jaxfne is missing required builder {builder_name!r}")
    builder = getattr(jtfne, builder_name)

    kwargs: dict[str, Any] = {
        "seed": spec.seed,
        "duration_ms": spec.duration_ms,
        "dt_ms": spec.dt_ms,
    }
    if spec.name == "suite2_single_neuron":
        kwargs["cell_type"] = spec.cell_type
    return builder(**kwargs)


def construct_jaxfne_model(
    spec: JaxfnePlaygroundSpec,
    *,
    jaxfne_module: Any | None = None,
) -> Any:
    """Construct a jaxfne model for the playground spec."""

    jtfne = jaxfne_module if jaxfne_module is not None else require_jaxfne()
    if not hasattr(jtfne, "construct"):
        raise AttributeError("jaxfne is missing required public API 'construct'")
    cfg = build_jaxfne_config(spec, jaxfne_module=jtfne)
    return jtfne.construct(cfg)


def simulate_jaxfne_model(
    model: Any,
    spec: JaxfnePlaygroundSpec,
    *,
    jaxfne_module: Any | None = None,
) -> Any:
    """Simulate a jaxfne model through the jaxfne public API."""

    spec.validate()
    jtfne = jaxfne_module if jaxfne_module is not None else require_jaxfne()
    if not hasattr(jtfne, "simulate"):
        raise AttributeError("jaxfne is missing required public API 'simulate'")

    kwargs: dict[str, Any] = {
        "duration_ms": spec.duration_ms,
        "dt_ms": spec.dt_ms,
        "seed": spec.seed,
    }
    if spec.noise_amplitude is not None:
        kwargs["noise_amplitude"] = spec.noise_amplitude
    return jtfne.simulate(model, **kwargs)


def _signal_get(signals: Any, name: str, **kwargs: Any) -> Any:
    getter = getattr(signals, "get", None)
    if callable(getter):
        return getter(name, **kwargs)
    fallback_names = {
        "vm": ("vm", "V_m", "voltage", "voltage_mV"),
        "spk": ("spk", "spikes", "spike", "spike_raster"),
    }
    for attr in fallback_names.get(name, (name,)):
        if hasattr(signals, attr):
            return getattr(signals, attr)
    raise AttributeError(f"signals object does not expose {name!r}")


def _finite_array(value: Any) -> bool:
    try:
        arr = np.asarray(value)
        if arr.size == 0:
            return False
        return bool(np.isfinite(arr).all())
    except Exception:
        return False


def validate_signal_contract(signals: Any, model: Any | None = None) -> dict[str, Any]:
    """Validate the minimal jaxfne Signals contract used by tutorials/workers."""

    vm = _signal_get(signals, "vm")
    spk = _signal_get(signals, "spk")
    vm_arr = np.asarray(vm)
    spk_arr = np.asarray(spk)
    errors: list[str] = []

    if vm_arr.ndim < 2:
        errors.append(f"vm must be at least 2D [time, unit], got shape {vm_arr.shape}")
    if spk_arr.ndim < 2:
        errors.append(f"spk must be at least 2D [time, unit], got shape {spk_arr.shape}")
    if vm_arr.shape != spk_arr.shape:
        errors.append(f"vm/spk shape mismatch: {vm_arr.shape} vs {spk_arr.shape}")
    if not _finite_array(vm_arr):
        errors.append("vm contains NaN/Inf or is empty")
    if not _finite_array(spk_arr):
        errors.append("spk contains NaN/Inf or is empty")

    e_selected = None
    e_vm_shape = None
    if model is not None and hasattr(model, "select"):
        try:
            e_selected = list(model.select(cell_type="E"))
            e_vm = _signal_get(signals, "vm", cell_type="E")
            e_vm_shape = tuple(np.asarray(e_vm).shape)
            if e_vm_shape[-1] != len(e_selected):
                errors.append(
                    "signals.get('vm', cell_type='E') shape does not match "
                    "model.select(cell_type='E')"
                )
        except Exception as exc:
            errors.append(f"E selector/signal smoke failed: {exc!r}")

    return json_safe(
        {
            "ok": len(errors) == 0,
            "errors": errors,
            "vm_shape": tuple(int(x) for x in vm_arr.shape),
            "spk_shape": tuple(int(x) for x in spk_arr.shape),
            "vm_all_finite": _finite_array(vm_arr),
            "spk_all_finite": _finite_array(spk_arr),
            "n_E_selected": len(e_selected) if e_selected is not None else None,
            "E_vm_shape": e_vm_shape,
        }
    )


def run_playground_smoke(
    spec: JaxfnePlaygroundSpec | None = None,
    *,
    jaxfne_module: Any | None = None,
) -> dict[str, Any]:
    """Construct, simulate, validate, and return a JSON-safe smoke receipt."""

    spec = spec or JaxfnePlaygroundSpec()
    spec.validate()
    backend_report = jaxfne_backend_report(jaxfne_module)
    jtfne = jaxfne_module if jaxfne_module is not None else require_jaxfne()

    model = construct_jaxfne_model(spec, jaxfne_module=jtfne)
    signals = simulate_jaxfne_model(model, spec, jaxfne_module=jtfne)
    validation = validate_signal_contract(signals, model)

    decision = "ACCEPT_SMOKE" if validation["ok"] else "REJECT_INVALID"
    return json_safe(
        {
            "kind": "jbiophysic_jaxfne_playground_smoke_receipt",
            "decision": decision,
            "spec": spec.to_json_dict(),
            "backend_report": backend_report,
            "validation": validation,
            "status_gates": build_request_manifest(spec)["status_gates"],
            "python": sys.version,
            "platform": platform.platform(),
        }
    )


def write_playground_receipt(receipt: dict[str, Any], path: str | Path) -> Path:
    """Write a strict JSON smoke/request receipt."""

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = json_safe(receipt)
    with out.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True, allow_nan=False)
        f.write("\n")
    return out
