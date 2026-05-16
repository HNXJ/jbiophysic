"""Machine-readable TFNE operator implementation status.

The mathematical TFNE manuscript defines an extended stack, but the repository only
implements selected developmental/scaffold pieces.  This table prevents accidental
claim inflation by making implemented, prototype, partial, and future operators explicit.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal

OperatorState = Literal[
    "repo_module",
    "partial_repo_module",
    "prototype_api",
    "specified_future_module",
    "not_implemented",
]


@dataclass(frozen=True)
class OperatorStatus:
    operator_id: str
    symbol: str
    name: str
    state: OperatorState
    implemented_paths: tuple[str, ...]
    prototype_paths: tuple[str, ...]
    missing_requirements: tuple[str, ...]
    claim_allowed: str
    claim_forbidden: str
    next_hardening_task: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def get_operator_status() -> dict[str, OperatorStatus]:
    """Return TFNE operator status keyed by public operator id."""
    return {
        "emitter": OperatorStatus(
            "emitter",
            "E_theta",
            "Emitter node operator",
            "repo_module",
            ("src/jbiophysic/cells", "src/jbiophysic/models/tfne_izhikevich.py"),
            ("src/jbiophysic/jtfne.py",),
            ("full HH/cable current export for all workflows",),
            "Reduced and conductance emitters can generate timing/native/source-proxy states.",
            "Reduced native drive is not physical amperes without calibration.",
            "Add emitter-level source-current provenance for every workflow.",
        ),
        "synapse": OperatorStatus(
            "synapse",
            "S_WDR",
            "Directed synapse/receptor edge operator",
            "partial_repo_module",
            ("src/jbiophysic/synapses", "src/jbiophysic/networks/connectivity.py"),
            ("src/jbiophysic/jtfne.py",),
            ("formal receptor-class tensor export", "delay/reversal/time-constant manifest export"),
            "Directed effective-weight scaffolds are available.",
            "Full receptor-class biophysical synapse tensor is not yet complete.",
            "Harden receptor-class edge tables and manifest serialization.",
        ),
        "chemical": OperatorStatus(
            "chemical",
            "C_mu_nu",
            "Chemical modulation operator",
            "specified_future_module",
            tuple(),
            tuple(),
            ("neurotransmitter fields", "neuromodulator fields", "electrodiffusion if claimed"),
            "Chemical variables may be specified as future parameter modulators.",
            "No q_chem electrical source-density claim in baseline TFNE.",
            "Implement parameter-modulation objects before chemical claims.",
        ),
        "source_projection": OperatorStatus(
            "source_projection",
            "Q_eta_alpha",
            "Source projection/calibration operator",
            "partial_repo_module",
            ("src/jbiophysic/tfne/sources.py", "src/jbiophysic/tfne/validation.py"),
            ("src/jbiophysic/jtfne.py",),
            ("exact-run per-kernel export", "empirical source-current calibration"),
            "Normalized kernels and source-sink compatibility are testable.",
            "Toy/proxy source scale is not empirical amplitude calibration.",
            "Export per-run kernel and integrated-current diagnostics.",
        ),
        "field": OperatorStatus(
            "field",
            "F_Omega_B_G_Gamma",
            "Tensor field solver operator",
            "partial_repo_module",
            (
                "src/jbiophysic/tfne/solvers.py",
                "src/jbiophysic/tfne/csd.py",
                "src/jbiophysic/tfne/tensors.py",
            ),
            tuple(),
            ("production sparse solver", "normalized residual for exact figure run"),
            "Smoke-scale resistive source-to-field solves are available.",
            "Smoke solver output is not calibrated empirical field evidence.",
            "Return normalized residual, boundary, gauge, and passivity diagnostics everywhere.",
        ),
        "probe": OperatorStatus(
            "probe",
            "P",
            "Probe/readout operator",
            "prototype_api",
            ("src/jbiophysic/analysis",),
            ("src/jbiophysic/jtfne.py", "src/jbiophysic/viz"),
            ("first-class finite-contact and interpolation probes",),
            "Developmental laminar contact readouts are available.",
            "EEG/MEG are proxy/future readouts unless head/sensor model is supplied.",
            "Add explicit probe geometry objects and tests.",
        ),
        "analysis": OperatorStatus(
            "analysis",
            "A",
            "Objective/evaluator operator",
            "partial_repo_module",
            ("src/jbiophysic/analysis", "src/jbiophysic/objectives"),
            ("src/jbiophysic/jtfne.py",),
            ("null-normalized S_lam with null distributions", "full diagnostics export"),
            "Internal motif-gate and profile metrics can be computed.",
            "Internal motif gate is not an empirical null-normalized similarity.",
            "Separate G_motif, profile score, and S_lam in outputs.",
        ),
        "optimizer": OperatorStatus(
            "optimizer",
            "O",
            "Optional optimizer feedback operator",
            "partial_repo_module",
            ("src/jbiophysic/optim",),
            ("src/jbiophysic/jtfne.py",),
            ("matched budget comparisons", "constraint projection manifest"),
            "Optimizer can search bounded scaffold parameters.",
            "Optimizer success is not biological proof.",
            "Unify differentiable vs black-box optimizer manifests.",
        ),
        "constraints": OperatorStatus(
            "constraints",
            "C",
            "Constraint/invariant validator",
            "partial_repo_module",
            ("src/jbiophysic/tfne/validation.py",),
            tuple(),
            ("centralized run-level rejection gates",),
            "Basic finite/source/passivity checks are available.",
            "Missing validator output must not be treated as passing evidence.",
            "Add composite JSON-safe validation reports.",
        ),
    }


def operator_status_json() -> dict[str, dict[str, object]]:
    """Return JSON-serializable operator status table keyed by repo role."""
    return {k: v.to_dict() for k, v in get_operator_status().items()}


def operator_status_by_symbol_json() -> dict[str, dict[str, object]]:
    """Return JSON-serializable operator status table keyed by formal TFNE symbol.

    Evidence artifacts use these keys because they correspond directly to the
    manuscript/operator doctrine: E_theta, S_WDR, C_mu_nu, Q_eta_alpha,
    F_field, P_probe, A_objective, O_optimizer, and C_constraints.
    """
    symbol_map = {
        "E_theta": "emitter",
        "S_WDR": "synapse",
        "C_mu_nu": "chemical",
        "Q_eta_alpha": "source_projection",
        "F_field": "field",
        "P_probe": "probe",
        "A_objective": "analysis",
        "O_optimizer": "optimizer",
        "C_constraints": "constraints",
    }
    roles = operator_status_json()
    out: dict[str, dict[str, object]] = {}
    for symbol, role in symbol_map.items():
        record = dict(roles[role])
        record["role_key"] = role
        record["operator_id"] = symbol
        out[symbol] = record
    return out
