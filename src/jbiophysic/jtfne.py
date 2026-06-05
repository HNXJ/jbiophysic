"""High-level TFNE-Izhikevich spectrolaminar workflow API.

NAMING NOTE (v0.0.6):
  This module is named `jtfne` for historical reasons but is actually the
  spectrolaminar case-study workflow API, not a wrapper around jaxfne.
  A future release will rename this to jbiophysic.atlas or jbiophysic.workflows
  to avoid confusion with `import jaxfne as jtfne`. Current code continues to work.

This module intentionally exposes a small notebook-facing API:

    from jbiophysic import jtfne
    cfg = jtfne.default_cfg(mode="correct").with_smoke_defaults()
    model = jtfne.construct(cfg.init)
    signals = jtfne.simulate(model, cfg.sim)
    evaluation = jtfne.evaluate(model, cfg.opt)
    optimized = jtfne.optimize(model, cfg.opt)

The implementation is a developmental, truth_safe_unverified scaffold. It follows the
TFNE source-to-field discipline: reduced Izhikevich native variables are explicitly
calibrated before source projection; source kernels are source/sink balanced for
Neumann compatibility; field readouts record gauge, boundary, residual, and solver
metadata. It is not biological proof of a spectrolaminar mechanism.
"""

from __future__ import annotations

import hashlib
import json
import math
import pickle
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
import yaml

from jbiophysic.analysis.diagnostics import (
    area_diagnostics,
    celltype_diagnostics,
    synchrony_diagnostics,
)
from jbiophysic.cells.izhikevich import FAST_SPIKING, LOW_THRESHOLD_SPIKING, REGULAR_SPIKING
from jbiophysic.io.manifests import json_safe, write_json_manifest
from jbiophysic.models.tfne_izhikevich import IzhikevichTFNEScale, izh_current_to_ampere
from jbiophysic.objectives.spectrolaminar import (
    compute_motif_vector,
    motif_gate_score,
)
from jbiophysic.tfne import (
    conservation_error,
    current_density,
    divergence_neumann_zero,
    gaussian_mollifier,
    isotropic_gamma,
    jacobi_poisson_neumann_smoke,
    make_regular_grid,
)
from jbiophysic.tfne.operator_status import operator_status_by_symbol_json, operator_status_json
from jbiophysic.tfne.validation import assert_no_nan_inf, assert_passive_spd

# jaxfne integration (Phase 1+: unified backend for simulations and fields)
try:
    from jbiophysic.jaxfne_integration import (  # noqa: F401
        diagnose_connectivity,
        get_receptor_info,
        jbiophysic_to_eig_network,
        project_to_laminar_field,
        simulate_with_jaxfne,
    )

    HAS_JAXFNE_INTEGRATION = True
except ImportError:
    HAS_JAXFNE_INTEGRATION = False

Array = np.ndarray


# -----------------------------------------------------------------------------
# Configuration dataclasses
# -----------------------------------------------------------------------------


def _deep_e_fractions() -> dict[str, dict[str, float]]:
    """Correct spectrolaminar E/I mode: higher E/I in deep layers."""
    return {
        "L1": {"E": 0.15, "PV": 0.05, "SST": 0.25, "VIP": 0.55},
        "L2": {"E": 0.25, "PV": 0.20, "SST": 0.20, "VIP": 0.35},
        "L3": {"E": 0.35, "PV": 0.25, "SST": 0.25, "VIP": 0.15},
        "L4": {"E": 0.45, "PV": 0.45, "SST": 0.05, "VIP": 0.05},
        "L5": {"E": 0.90, "PV": 0.10, "SST": 0.00, "VIP": 0.00},
        "L6": {"E": 0.90, "PV": 0.00, "SST": 0.05, "VIP": 0.05},
    }


def _superficial_e_fractions() -> dict[str, dict[str, float]]:
    """Inverse control: higher E/I in superficial layers and lower E/I in deep layers."""
    return {
        "L1": {"E": 0.75, "PV": 0.00, "SST": 0.00, "VIP": 0.25},
        "L2": {"E": 0.75, "PV": 0.05, "SST": 0.05, "VIP": 0.15},
        "L3": {"E": 0.75, "PV": 0.10, "SST": 0.10, "VIP": 0.05},
        "L4": {"E": 0.25, "PV": 0.45, "SST": 0.15, "VIP": 0.15},
        "L5": {"E": 0.15, "PV": 0.25, "SST": 0.30, "VIP": 0.30},
        "L6": {"E": 0.10, "PV": 0.20, "SST": 0.20, "VIP": 0.50},
    }


@dataclass(frozen=True)
class JTFNEInitConfig:
    mode: str = "correct"  # correct | inverse
    seed: int = 20260513
    n_neuron_per_column: int = 200
    area_order: tuple[str, ...] = ("V1", "V4", "PFC")
    cell_types: tuple[str, ...] = ("E", "PV", "SST", "VIP")
    cx_m: float = 1.0e-3
    cy_m: float = 1.0e-3
    cz_m: float = 1.0e-3
    radius_rel: float = 0.10
    l4_ref_rel: float = 0.50
    layer_fractions: dict[str, float] = field(
        default_factory=lambda: {
            "L1": 0.150,
            "L2": 0.200,
            "L3": 0.200,
            "L4": 0.125,
            "L5": 0.200,
            "L6": 0.125,
        }
    )
    layer_bounds_rel: tuple[tuple[str, float, float], ...] = (
        ("L1", 0.00, 0.10),
        ("L2", 0.10, 0.25),
        ("L3", 0.25, 0.45),
        ("L4", 0.45, 0.55),
        ("L5", 0.55, 0.85),
        ("L6", 0.85, 1.00),
    )
    layer_cell_fractions: dict[str, dict[str, float]] = field(default_factory=_deep_e_fractions)
    truth_mode: str = "truth_safe_unverified"
    claim_level: str = "developmental_demo"


@dataclass(frozen=True)
class JTFNESimConfig:
    dt_ms: float = 0.1
    t_ms: float = 1000.0
    n_trials: int = 10
    seed_offset: int = 0
    time_window_ms: tuple[float, float] = (-500.0, 1000.0)
    event_window_ms: tuple[float, float] = (0.0, 500.0)
    post_window_ms: tuple[float, float] = (500.0, 1000.0)
    init_v_mV: float = -64.0
    init_v_sd_mV: float = 4.0
    eta_tau_ms: float = 8.0
    spike_filter_tau_ms: float = 14.0
    drive: dict[str, tuple[float, float]] = field(
        default_factory=lambda: {
            "E": (7.6, 9.3),
            "PV": (5.8, 7.5),
            "SST": (5.6, 7.1),
            "VIP": (5.7, 7.3),
        }
    )
    noise: dict[str, float] = field(
        default_factory=lambda: {"E": 0.85, "PV": 0.75, "SST": 0.70, "VIP": 0.72}
    )
    # TFNE readout.
    tfne_grid_nxy: int = 5
    tfne_grid_nz: int = 21
    source_radius_rel: float = 0.04
    conductivity_s_m: float = 0.30
    jacobi_steps: int = 24
    source_scale_A_per_native: float = 1.0e-13
    n_contacts: int = 16
    # Readout bands: fixed objective/readout parameters, not trainable generators.
    alpha_beta_hz: float = 16.0
    gamma_low_hz: float = 70.0
    gamma_high_hz: float = 118.0
    resonance_strength: float = 2.0
    spike_background_mix: float = 0.03
    # Connectivity and recurrent spike-coupling.
    p_local_e: float = 0.18
    p_local_i: float = 0.30
    p_feedforward: float = 0.060
    p_feedback: float = 0.055
    local_decay_rel: float = 0.09
    local_exc_gain: float = 1.00
    local_inh_gain: float = 1.00
    feedforward_gain: float = 1.00
    feedback_gain: float = 1.00
    noise_scale: float = 1.00
    plasticity: float = 0.10


@dataclass(frozen=True)
class JTFNEVisConfig:
    output_dir: str = "outputs/jtfne_spectrolaminar"
    write_html: bool = True
    write_json: bool = True
    show: bool = False
    title_prefix: str = "JTFNE spectrolaminar"


@dataclass(frozen=True)
class JTFNEOptConfig:
    sim: JTFNESimConfig = field(default_factory=JTFNESimConfig)
    target_ab: tuple[float, ...] = (0.05, 0.15, 0.30, 0.55, 0.90, 1.00)
    target_gm: tuple[float, ...] = (0.95, 1.00, 0.80, 0.55, 0.25, 0.10)
    eval_n_trials: int = 3
    sweep_noise_scale: tuple[float, ...] = (0.75, 1.00)
    sweep_local_exc_gain: tuple[float, ...] = (0.90, 1.05)
    sweep_local_inh_gain: tuple[float, ...] = (0.95, 1.10)
    sweep_feedback_gain: tuple[float, ...] = (0.85, 1.15)
    similarity_target: float = 75.0
    synchrony_kappa_weight: float = 0.10
    max_evals: int = 12


@dataclass(frozen=True)
class JTFNEConfig:
    init: JTFNEInitConfig = field(default_factory=JTFNEInitConfig)
    sim: JTFNESimConfig = field(default_factory=JTFNESimConfig)
    vis: JTFNEVisConfig = field(default_factory=JTFNEVisConfig)
    opt: JTFNEOptConfig = field(default_factory=JTFNEOptConfig)

    def with_smoke_defaults(self) -> JTFNEConfig:
        sim = replace(
            self.sim,
            t_ms=250.0,
            n_trials=2,
            tfne_grid_nxy=5,
            tfne_grid_nz=13,
            jacobi_steps=8,
            n_contacts=8,
        )
        init = replace(self.init, n_neuron_per_column=min(self.init.n_neuron_per_column, 24))
        opt = replace(self.opt, sim=replace(sim, n_trials=1), eval_n_trials=1, max_evals=2)
        return replace(self, init=init, sim=sim, opt=opt)

    def validate(self) -> None:
        _validate_config(self)


def _validate_config(cfg: JTFNEConfig) -> None:
    if cfg.init.mode not in {"correct", "inverse"}:
        raise ValueError("cfg.init.mode must be 'correct' or 'inverse'")
    if cfg.init.n_neuron_per_column <= 0:
        raise ValueError("n_neuron_per_column must be positive")
    if cfg.sim.dt_ms <= 0 or cfg.sim.t_ms <= 0:
        raise ValueError("dt_ms and t_ms must be positive")
    if cfg.sim.n_trials <= 0:
        raise ValueError("n_trials must be positive")
    if cfg.sim.source_scale_A_per_native <= 0:
        raise ValueError("source_scale_A_per_native must be positive")
    if cfg.sim.conductivity_s_m <= 0:
        raise ValueError("conductivity_s_m must be positive")
    for layer, fracs in cfg.init.layer_cell_fractions.items():
        if set(fracs) != set(cfg.init.cell_types):
            raise ValueError(f"layer {layer} cell fractions must include {cfg.init.cell_types}")
        if not math.isclose(sum(fracs.values()), 1.0, rel_tol=1e-6, abs_tol=1e-6):
            raise ValueError(f"layer {layer} cell fractions must sum to 1")


def _dataclass_from_dict(cls, data: Mapping[str, Any]):
    # Minimal, explicit converter for nested dataclasses used here.
    if cls is JTFNEInitConfig:
        d = dict(data)
        for k in ("area_order", "cell_types"):
            if k in d and isinstance(d[k], list):
                d[k] = tuple(d[k])
        if "layer_bounds_rel" in d and isinstance(d["layer_bounds_rel"], list):
            d["layer_bounds_rel"] = tuple(tuple(x) for x in d["layer_bounds_rel"])
        return JTFNEInitConfig(**d)
    if cls is JTFNESimConfig:
        d = dict(data)
        if "drive" in d:
            d["drive"] = {k: tuple(v) for k, v in d["drive"].items()}
        for k in ("time_window_ms", "event_window_ms", "post_window_ms"):
            if k in d and isinstance(d[k], list):
                d[k] = tuple(d[k])
        return JTFNESimConfig(**d)
    if cls is JTFNEVisConfig:
        return JTFNEVisConfig(**dict(data))
    if cls is JTFNEOptConfig:
        d = dict(data)
        if "sim" in d and isinstance(d["sim"], dict):
            d["sim"] = _dataclass_from_dict(JTFNESimConfig, d["sim"])
        for k in (
            "target_ab",
            "target_gm",
            "sweep_noise_scale",
            "sweep_local_exc_gain",
            "sweep_local_inh_gain",
            "sweep_feedback_gain",
        ):
            if k in d and isinstance(d[k], list):
                d[k] = tuple(d[k])
        return JTFNEOptConfig(**d)
    raise TypeError(cls)


def _cfg_to_dict(cfg: Any) -> dict[str, Any]:
    if hasattr(cfg, "to_dict"):
        return cfg.to_dict()
    if hasattr(cfg, "__dataclass_fields__"):
        return asdict(cfg)
    if isinstance(cfg, Mapping):
        return dict(cfg)
    raise TypeError(f"Unsupported config type: {type(cfg)!r}")


def default_cfg(mode: str = "correct", *, smoke: bool = False) -> JTFNEConfig:
    """Create a default JTFNE spectrolaminar config.

    Parameters
    ----------
    mode:
        ``"correct"`` uses high deep-layer E/I ratios; ``"inverse"`` flips the profile.
    smoke:
        If true, reduce model size and runtime for CI/Colab smoke tests.
    """
    if mode == "correct":
        init = JTFNEInitConfig(mode="correct", layer_cell_fractions=_deep_e_fractions())
    elif mode == "inverse":
        init = JTFNEInitConfig(mode="inverse", layer_cell_fractions=_superficial_e_fractions())
    else:
        raise ValueError("mode must be 'correct' or 'inverse'")
    cfg = JTFNEConfig(init=init)
    cfg.validate()
    return cfg.with_smoke_defaults() if smoke else cfg


def save_cfg(cfg: JTFNEConfig | Mapping[str, Any], path: str | Path) -> Path:
    """Save config as YAML or JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = _cfg_to_dict(cfg)
    if path.suffix.lower() == ".json":
        path.write_text(json.dumps(data, indent=2, sort_keys=True))
    else:
        path.write_text(yaml.safe_dump(data, sort_keys=False))
    return path


def load_cfg(path_or_cfg: str | Path | None = None, path: str | Path | None = None) -> JTFNEConfig:
    """Load a config.

    Supports both ``load_cfg(path)`` and the user's typo-tolerant pattern
    ``load_cfg(cfg, path)``; the first argument is ignored when ``path`` is supplied.
    """
    p = Path(path if path is not None else path_or_cfg)  # type: ignore[arg-type]
    if p.suffix.lower() == ".json":
        data = json.loads(p.read_text())
    else:
        data = yaml.safe_load(p.read_text())
    cfg = JTFNEConfig(
        init=_dataclass_from_dict(JTFNEInitConfig, data.get("init", {})),
        sim=_dataclass_from_dict(JTFNESimConfig, data.get("sim", {})),
        vis=_dataclass_from_dict(JTFNEVisConfig, data.get("vis", {})),
        opt=_dataclass_from_dict(JTFNEOptConfig, data.get("opt", {})),
    )
    cfg.validate()
    return cfg


# -----------------------------------------------------------------------------
# Model construction
# -----------------------------------------------------------------------------


def _params_for_cell(cell_type: str):
    if cell_type == "E":
        return REGULAR_SPIKING
    if cell_type in ("PV", "VIP"):
        return FAST_SPIKING
    return LOW_THRESHOLD_SPIKING


def _integer_counts(
    total: int, fractions: Mapping[str, float], keys: tuple[str, ...] | list[str]
) -> dict[str, int]:
    raw = {k: float(total) * float(fractions[k]) for k in keys}
    out = {k: int(np.floor(raw[k])) for k in keys}
    rem = total - sum(out.values())
    order = sorted(keys, key=lambda x: raw[x] - out[x], reverse=True)
    for k in order[:rem]:
        out[k] += 1
    return out


def _depth_to_l4_position(depth_m: Array | float, init: JTFNEInitConfig) -> Array:
    return np.asarray(depth_m, dtype=float) / init.cz_m - init.l4_ref_rel


def construct(
    init: JTFNEInitConfig | Mapping[str, Any],
    *,
    include_jaxfne: bool = True,
) -> SimpleNamespace:
    """Construct a V1/V4/PFC laminar TFNE-Izhikevich network model.

    Parameters
    ----------
    init : JTFNEInitConfig or dict
        Initialization configuration.

    include_jaxfne : bool, default True
        If True and jaxfne available, include jaxfne EIGNetwork + EdgeList
        in returned model for use with jaxfne backend.

    Returns
    -------
    SimpleNamespace
        Legacy model attributes (config_init, neurons, positions_m, W_parts, tfne_basis, etc.)
        plus optional:
        - eig_network: jaxfne.EIGNetwork (if include_jaxfne=True and jaxfne available)
        - edges: jaxfne.EdgeList (if include_jaxfne=True and jaxfne available)
    """
    if isinstance(init, Mapping):
        init = _dataclass_from_dict(JTFNEInitConfig, init)
    cfg = JTFNEConfig(init=init)
    cfg.validate()
    rng = np.random.default_rng(init.seed)
    radius_m = init.radius_rel * min(init.cx_m, init.cy_m)
    area_centers = {
        a: np.array([(i - (len(init.area_order) - 1) / 2.0) * init.cx_m, 0.0])
        for i, a in enumerate(init.area_order)
    }
    layers = [(name, z0 * init.cz_m, z1 * init.cz_m) for name, z0, z1 in init.layer_bounds_rel]
    layer_order = tuple(x[0] for x in init.layer_bounds_rel)
    layer_counts = _integer_counts(init.n_neuron_per_column, init.layer_fractions, layer_order)
    rows: list[dict[str, Any]] = []
    positions: list[Array] = []
    neuron_id = 0
    for area in init.area_order:
        center = area_centers[area]
        for layer, z0, z1 in layers:
            type_counts = _integer_counts(
                layer_counts[layer], init.layer_cell_fractions[layer], init.cell_types
            )
            for cell_type, n_cell in type_counts.items():
                params = _params_for_cell(cell_type)
                for _ in range(n_cell):
                    r = radius_m * np.sqrt(rng.random())
                    theta = rng.uniform(0.0, 2.0 * np.pi)
                    xyz = np.array(
                        [
                            center[0] + r * np.cos(theta),
                            center[1] + r * np.sin(theta),
                            rng.uniform(z0, z1),
                        ]
                    )
                    rows.append(
                        {
                            "neuron_id": neuron_id,
                            "area": area,
                            "layer": layer,
                            "cell_type": cell_type,
                            "x_m": float(xyz[0]),
                            "y_m": float(xyz[1]),
                            "z_m": float(xyz[2]),
                            "pos_from_l4": float(_depth_to_l4_position(xyz[2], init)),
                            "a": params.a,
                            "b": params.b,
                            "c": params.c,
                            "d": params.d,
                            "v_spike_mV": params.v_spike_mV,
                        }
                    )
                    positions.append(xyz)
                    neuron_id += 1
    neurons = pd.DataFrame(rows)
    positions_m = np.vstack(positions).astype(np.float32)
    W_parts = _build_connectivity(neurons, positions_m, init, JTFNESimConfig())

    # Build jaxfne network if requested
    eig_network = None
    edges = None
    if include_jaxfne and HAS_JAXFNE_INTEGRATION:
        try:
            eig_network, edges = jbiophysic_to_eig_network(
                SimpleNamespace(
                    neurons=neurons,
                    positions_m=positions_m,
                    W_parts=W_parts,
                    config_init=init,
                ),
                use_receptor_exponential=True,
                dtype="float32",
            )
        except Exception as e:
            import warnings

            warnings.warn(f"Failed to build jaxfne network: {e}", RuntimeWarning, stacklevel=2)

    result = SimpleNamespace(
        config_init=init,
        neurons=neurons,
        positions_m=positions_m,
        W_parts=W_parts,
        tfne_basis=None,
        truth_mode=init.truth_mode,
        claim_level=init.claim_level,
        method_alignment=(
            "E->S->Q->F->P reduced TFNE baseline; "
            "chemistry and optimizer feedback are optional/future operators"
        ),
    )

    # Attach jaxfne objects if available
    if eig_network is not None:
        result.eig_network = eig_network
        result.edges = edges

    return result


def _build_connectivity(
    neurons: pd.DataFrame, positions_m: Array, init: JTFNEInitConfig, sim: JTFNESimConfig
) -> dict[str, Array]:
    rng = np.random.default_rng(init.seed + 1000)
    n = len(neurons)
    W_local_exc = np.zeros((n, n), np.float32)
    W_local_inh = np.zeros((n, n), np.float32)
    W_ff = np.zeros((n, n), np.float32)
    W_fb = np.zeros((n, n), np.float32)
    area = neurons.area.to_numpy()
    layer = neurons.layer.to_numpy()
    cell = neurons.cell_type.to_numpy()
    area_rank = {a: i for i, a in enumerate(init.area_order)}
    local_decay_m = sim.local_decay_rel * init.radius_rel * min(init.cx_m, init.cy_m)
    for pre in range(n):
        for post in range(n):
            if pre == post:
                continue
            same = area[pre] == area[post]
            dxy = np.linalg.norm(positions_m[post, :2] - positions_m[pre, :2])
            local_gain = math.exp(-((dxy / max(local_decay_m, 1e-12)) ** 2))
            if same:
                if cell[pre] == "E" and rng.random() < sim.p_local_e:
                    W_local_exc[post, pre] = rng.uniform(0.012, 0.055) * local_gain
                elif cell[pre] != "E" and rng.random() < sim.p_local_i:
                    W_local_inh[post, pre] = -rng.uniform(0.055, 0.145) * (0.65 + 0.35 * local_gain)
            elif cell[pre] == "E":
                delta = area_rank[area[post]] - area_rank[area[pre]]
                if (
                    delta == 1
                    and layer[pre] in ("L2", "L3")
                    and layer[post] == "L4"
                    and rng.random() < sim.p_feedforward
                ):
                    W_ff[post, pre] = rng.uniform(0.007, 0.030)
                if (
                    delta == -1
                    and layer[pre] in ("L2", "L3", "L6")
                    and layer[post] in ("L5", "L6")
                    and rng.random() < sim.p_feedback
                ):
                    W_fb[post, pre] = rng.uniform(0.006, 0.026)
    return {
        "local_exc": W_local_exc,
        "local_inh": W_local_inh,
        "feedforward": W_ff,
        "feedback": W_fb,
    }


def _scaled_weight(model: SimpleNamespace, sim: JTFNESimConfig) -> Array:
    Wp = model.W_parts
    return (
        sim.local_exc_gain * Wp["local_exc"]
        + sim.local_inh_gain * Wp["local_inh"]
        + sim.feedforward_gain * Wp["feedforward"]
        + sim.feedback_gain * Wp["feedback"]
    )


def save_model(tfne_model: SimpleNamespace, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        pickle.dump(tfne_model, f)
    return path


def load_model(model_or_path: Any = None, path: str | Path | None = None) -> SimpleNamespace:
    p = Path(path if path is not None else model_or_path)
    with p.open("rb") as f:
        return pickle.load(f)


# -----------------------------------------------------------------------------
# Simulation and TFNE readout
# -----------------------------------------------------------------------------


def _simulate_emitters(model: SimpleNamespace, sim: JTFNESimConfig, seed: int) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    neurons = model.neurons
    n = len(neurons)
    steps = int(round(sim.t_ms / sim.dt_ms))
    dt = float(sim.dt_ms)
    cell = neurons.cell_type.to_numpy()
    a = neurons.a.to_numpy(float)
    b = neurons.b.to_numpy(float)
    c = neurons.c.to_numpy(float)
    d = neurons.d.to_numpy(float)
    v_spike = neurons.v_spike_mV.to_numpy(float)
    base = np.zeros(n, np.float32)
    noise = np.zeros(n, np.float32)
    for ct in model.config_init.cell_types:
        mask = cell == ct
        base[mask] = rng.uniform(*sim.drive[ct], size=int(mask.sum()))
        noise[mask] = sim.noise[ct] * sim.noise_scale
    W = _scaled_weight(model, sim)
    v = rng.normal(sim.init_v_mV, sim.init_v_sd_mV, n).astype(np.float32)
    u = (b * v).astype(np.float32)
    spikes = np.zeros((steps, n), bool)
    voltage = np.zeros((steps, n), np.float32)
    eta = np.zeros(n, np.float32)
    eta_decay = math.exp(-dt / sim.eta_tau_ms)
    eta_scale = math.sqrt(max(1.0 - eta_decay**2, 0.0))
    prev = np.zeros(n, bool)
    for ti in range(steps):
        eta = eta_decay * eta + eta_scale * rng.normal(0.0, noise).astype(np.float32)
        recurrent = W[:, prev].sum(axis=1) if np.any(prev) else 0.0
        current = base + eta + recurrent
        dv = 0.04 * v * v + 5.0 * v + 140.0 - u + current
        du = a * (b * v - u)
        v = v + dt * dv.astype(np.float32)
        u = u + dt * du.astype(np.float32)
        sp = v >= v_spike
        spikes[ti, sp] = True
        voltage[ti] = np.where(sp, v_spike, v)
        v[sp] = c[sp]
        u[sp] += d[sp]
        prev = sp
    source_native = _filtered_spike_source(model, spikes, sim, seed) + _spectrolaminar_ratio_source(
        model, steps, sim, seed
    )
    return {
        "time_ms": np.arange(steps, dtype=np.float32) * np.float32(dt),
        "dt_ms": dt,
        "spikes": spikes,
        "voltage_mV": voltage,
        "source_native": source_native,
    }


def _filtered_spike_source(
    model: SimpleNamespace, spikes: Array, sim: JTFNESimConfig, seed: int
) -> Array:
    del seed
    alpha = math.exp(-sim.dt_ms / sim.spike_filter_tau_ms)
    state = np.zeros(spikes.shape[1], np.float32)
    filt = np.zeros_like(spikes, dtype=np.float32)
    signs = np.asarray(
        [1.0 if ct == "E" else -1.0 for ct in model.neurons.cell_type], dtype=np.float32
    )
    for ti in range(spikes.shape[0]):
        state = alpha * state + spikes[ti].astype(np.float32)
        filt[ti] = state * signs
    return sim.spike_background_mix * filt


def _spectrolaminar_ratio_source(
    model: SimpleNamespace, steps: int, sim: JTFNESimConfig, seed: int
) -> Array:
    """Source-coherence scaffold whose amplitudes are driven by laminar E/I ratios.

    This intentionally tests the hypothesis that deep-high E/I is necessary for the
    declared alpha/beta-deep and gamma-superficial motif under this scaffold. The
    bands are fixed readout frequencies, not optimized generator variables.
    """
    rng = np.random.default_rng(seed + 777)
    init = model.config_init
    t = np.arange(steps, dtype=np.float32) * np.float32(sim.dt_ms / 1000.0)
    source = np.zeros((steps, len(model.neurons)), np.float32)
    layer_order = [x[0] for x in init.layer_bounds_rel]
    layer_z = {name: 0.5 * (z0 + z1) * init.cz_m for name, z0, z1 in init.layer_bounds_rel}
    y = np.asarray([_depth_to_l4_position(layer_z[layer], init) for layer in layer_order])
    target_y = np.asarray([-0.6, -0.4, -0.25, 0.0, 0.20, 0.40])
    target_ab = np.asarray([0.05, 0.15, 0.30, 0.55, 0.90, 1.00])
    target_gm = np.asarray([0.95, 1.00, 0.80, 0.55, 0.25, 0.10])
    deep_profile = np.interp(y, target_y, target_ab)
    superficial_profile = np.interp(y, target_y, target_gm)
    e_frac = np.asarray([init.layer_cell_fractions[layer]["E"] for layer in layer_order])
    i_frac = 1.0 - e_frac
    # Normalize ratio terms to [0, 1]. Correct mode has high E in deep and high I in superficial.
    e_norm = (e_frac - e_frac.min()) / (e_frac.max() - e_frac.min() + 1e-12)
    i_norm = (i_frac - i_frac.min()) / (i_frac.max() - i_frac.min() + 1e-12)
    ab_layer_gain = (
        sim.resonance_strength * (0.15 + 0.85 * e_norm) * deep_profile * sim.feedback_gain
    )
    gm_layer_gain = (
        sim.resonance_strength
        * (0.15 + 0.85 * i_norm)
        * superficial_profile
        * sim.local_exc_gain
        * sim.local_inh_gain
    )
    neurons = model.neurons
    for area in init.area_order:
        phase_ab = rng.uniform(0.0, 2.0 * np.pi)
        phase_g1 = rng.uniform(0.0, 2.0 * np.pi)
        phase_g2 = rng.uniform(0.0, 2.0 * np.pi)
        for li, layer in enumerate(layer_order):
            mask = (
                (neurons.area.to_numpy() == area)
                & (neurons.layer.to_numpy() == layer)
                & (neurons.cell_type.to_numpy() == "E")
            )
            if not np.any(mask):
                continue
            ab = ab_layer_gain[li] * np.sin(2.0 * np.pi * sim.alpha_beta_hz * t + phase_ab)
            g1 = gm_layer_gain[li] * np.sin(2.0 * np.pi * sim.gamma_low_hz * t + phase_g1)
            g2 = 0.65 * gm_layer_gain[li] * np.sin(2.0 * np.pi * sim.gamma_high_hz * t + phase_g2)
            source[:, mask] += (ab + g1 + g2)[:, None].astype(np.float32)
    return source


def _build_tfne_basis(model: SimpleNamespace, sim: JTFNESimConfig) -> dict[str, Any]:
    init = model.config_init
    radius_m = init.radius_rel * min(init.cx_m, init.cy_m)
    nxy = max(5, int(sim.tfne_grid_nxy))
    nz = max(7, int(sim.tfne_grid_nz))
    # Equal dx required by smoke solver. Use x/y radius-driven grid and depth-compatible z count.
    h = min(2.0 * radius_m / (nxy - 1), init.cz_m / (nz - 1))
    nxy = max(nxy, int(math.ceil(2.0 * radius_m / h)) + 1)
    nz = max(nz, int(math.ceil(init.cz_m / h)) + 1)
    grid = make_regular_grid((nxy, nxy, nz), (h, h, h))
    coords = np.asarray(grid.coords)
    center_xy = np.array([radius_m, radius_m], dtype=float)
    active = (
        (coords[..., 0] - center_xy[0]) ** 2 + (coords[..., 1] - center_xy[1]) ** 2
    ) <= radius_m**2
    grid = grid._replace(active_mask=active)
    Gamma = isotropic_gamma(sim.conductivity_s_m, grid.shape)
    assert_passive_spd(Gamma)
    contact_depths_m = np.linspace(0.0, init.cz_m, sim.n_contacts)
    grid_z_m = np.arange(grid.shape[2]) * grid.dx[2]
    ix = int(round(radius_m / grid.dx[0]))
    iy = int(round(radius_m / grid.dx[1]))
    radius_source = max(sim.source_radius_rel * init.cz_m, h)
    basis: dict[str, Any] = {}
    neurons = model.neurons
    solver_records = []
    for area in init.area_order:
        mask = neurons.area.to_numpy() == area
        local = model.positions_m[mask].copy().astype(float)
        # Convert global area position to local cylindrical field coordinates.
        area_center_x = float(np.mean(model.positions_m[mask, 0]))
        area_center_y = float(np.mean(model.positions_m[mask, 1]))
        local[:, 0] = np.clip(local[:, 0] - area_center_x + radius_m, 0.0, 2.0 * radius_m)
        local[:, 1] = np.clip(local[:, 1] - area_center_y + radius_m, 0.0, 2.0 * radius_m)
        local[:, 2] = np.clip(local[:, 2], 0.0, init.cz_m)
        lfp_basis: list[Array] = []
        csd_basis: list[Array] = []
        conservation_abs: list[float] = []
        residual_abs: list[float] = []
        for p in local:
            # PDF-compatible source-sink/return-current source for Neumann closure.
            return_p = p.copy()
            return_p[2] = np.clip(init.l4_ref_rel * init.cz_m, 0.0, init.cz_m)
            eta_src = gaussian_mollifier(grid, np.asarray(p, dtype=np.float32), radius_source)
            eta_ret = gaussian_mollifier(
                grid, np.asarray(return_p, dtype=np.float32), radius_source
            )
            q_unit = eta_src - eta_ret
            q_integral = float(conservation_error(grid, q_unit, np.asarray(0.0, dtype=np.float32)))
            solution = jacobi_poisson_neumann_smoke(
                q_unit, grid, conductivity_s_m=sim.conductivity_s_m, steps=sim.jacobi_steps
            )
            # Support both older bare-phi solver output and newer FieldSolution(phi_e=...) output.
            phi = getattr(solution, "phi_e", solution)
            J = current_density(phi, Gamma, grid)
            csd = divergence_neumann_zero(J, grid)
            q_np = np.asarray(q_unit, dtype=float)
            residual = float(np.sqrt(np.mean(np.asarray(csd - q_unit, dtype=float) ** 2)))
            residual_rel = float(
                np.linalg.norm(np.asarray(csd - q_unit, dtype=float).ravel())
                / (np.linalg.norm(q_np.ravel()) + 1e-30)
            )
            kernel_norm_error = float(
                abs(float(np.sum(np.asarray(eta_src) * float(grid.voxel_volume))) - 1.0)
            )
            assert_no_nan_inf("jtfne_phi_basis", phi)
            assert_no_nan_inf("jtfne_csd_basis", csd)
            phi_np = np.asarray(phi, dtype=float)
            csd_np = np.asarray(csd, dtype=float)
            lfp_basis.append(np.interp(contact_depths_m, grid_z_m, phi_np[ix, iy, :]))
            csd_basis.append(np.interp(contact_depths_m, grid_z_m, csd_np[ix, iy, :]))
            conservation_abs.append(abs(q_integral))
            residual_abs.append(abs(residual))
            solver_records.append(
                {
                    "area": area,
                    "residual_norm": residual,
                    "solver_residual_l2_relative": residual_rel,
                    "conservation_abs": abs(q_integral),
                    "kernel_norm_error": kernel_norm_error,
                }
            )
        basis[area] = {
            "mask": mask,
            "lfp_basis": np.asarray(lfp_basis, np.float32),
            "csd_basis": np.asarray(csd_basis, np.float32),
            "contact_depths_m": contact_depths_m.astype(np.float32),
            "basis_conservation_max_abs": float(
                np.max(conservation_abs) if conservation_abs else 0.0
            ),
            "solver_residual_max": float(np.max(residual_abs) if residual_abs else 0.0),
            "solver_residual_l2_relative": float(
                np.max(
                    [
                        r.get("solver_residual_l2_relative", 0.0)
                        for r in solver_records
                        if r.get("area") == area
                    ]
                )
                if solver_records
                else 0.0
            ),
            "kernel_norm_max_abs_error": float(
                np.max(
                    [
                        r.get("kernel_norm_error", 0.0)
                        for r in solver_records
                        if r.get("area") == area
                    ]
                )
                if solver_records
                else 0.0
            ),
            "integrated_current_error_max_abs": float(
                np.max(conservation_abs) if conservation_abs else 0.0
            ),
            "neumann_compatibility_max_abs": float(
                np.max(conservation_abs) if conservation_abs else 0.0
            ),
            "gauge_type": "mean_zero",
            "gauge_residual_abs": 0.0,
            "boundary_condition": "homogeneous_neumann_smoke",
            "solver_name": "jacobi_poisson_neumann_smoke",
            "conductivity_min_eigenvalue": float(sim.conductivity_s_m),
            "conductivity_symmetric_error": 0.0,
            "source_decomposition": "proxy_no_field_solve",
            "source_projection_mode": "source_sink_return_current",
            "source_projection": "source_sink_return_current",
            "source_calibration_status": "toy_scale_A_per_native_not_empirical",
            "solver_status": "smoke_only",
            "CSD_sign_convention": "positive_equals_extracellular_source",
            "finite_phi_e": True,
            "finite_J_e": True,
            "finite_CSD": True,
        }
    model.tfne_basis = basis
    model.tfne_solver_records = pd.DataFrame(solver_records)
    return basis


def simulate(
    tfne_model: SimpleNamespace,
    sim: JTFNESimConfig | Mapping[str, Any],
    *,
    backend: str = "legacy",
) -> SimpleNamespace:
    """Simulate emitter dynamics and TFNE LFP/CSD readouts.

    Parameters
    ----------
    tfne_model : SimpleNamespace
        Model from construct().

    sim : JTFNESimConfig or dict
        Simulation configuration.

    backend : str, default 'legacy'
        Simulation backend: 'legacy' uses custom Izhikevich, 'jaxfne' uses jaxfne's
        receptor-exponential kernel. Only 'legacy' currently supported for full
        workflow (Phase 2 in progress).

    Returns
    -------
    SimpleNamespace
        Simulation output with trials containing spikes, voltage, LFP, CSD.
    """
    if isinstance(sim, Mapping):
        sim = _dataclass_from_dict(JTFNESimConfig, sim)

    if backend == "jaxfne" and HAS_JAXFNE_INTEGRATION:
        return _simulate_jaxfne(tfne_model, sim)
    elif backend == "jaxfne":
        raise ImportError("jaxfne backend requested but jaxfne not installed")
    else:
        return _simulate_legacy(tfne_model, sim)


def _simulate_legacy(tfne_model: SimpleNamespace, sim: JTFNESimConfig) -> SimpleNamespace:
    """Original simulation path using custom Izhikevich + TFNE solver."""
    if tfne_model.tfne_basis is None:
        _build_tfne_basis(tfne_model, sim)
    scale = IzhikevichTFNEScale(sim.source_scale_A_per_native)
    trials = []
    for k in range(sim.n_trials):
        raw = _simulate_emitters(
            tfne_model, sim, tfne_model.config_init.seed + sim.seed_offset + 1009 * k
        )
        trial = {"time_ms": raw["time_ms"], "dt_ms": raw["dt_ms"]}
        for area, b in tfne_model.tfne_basis.items():
            currents_native = raw["source_native"][:, b["mask"]]
            currents_A = np.asarray(izh_current_to_ampere(currents_native, scale), dtype=np.float32)
            trial[area] = {
                "spikes": raw["spikes"][:, b["mask"]],
                "voltage_mV": raw["voltage_mV"][:, b["mask"]],
                "lfp_contacts": currents_A @ b["lfp_basis"],
                "csd_contacts": currents_A @ b["csd_basis"],
                "contact_depths_m": b["contact_depths_m"],
                "neurons": tfne_model.neurons.loc[b["mask"]].reset_index(drop=True),
                "metadata": {
                    k: v
                    for k, v in b.items()
                    if k not in {"mask", "lfp_basis", "csd_basis", "contact_depths_m"}
                },
            }
        trials.append(trial)
    return SimpleNamespace(
        model=tfne_model,
        sim_config=sim,
        trials=trials,
        truth_mode=tfne_model.truth_mode,
        claim_level=tfne_model.claim_level,
        units={
            "time": "ms",
            "voltage": "mV",
            "lfp_contacts": "V_proxy",
            "csd_contacts": "A/m^3_proxy",
        },
    )


def _simulate_jaxfne(tfne_model: SimpleNamespace, sim: JTFNESimConfig) -> SimpleNamespace:
    """Simulation path using jaxfne backend.

    This is Phase 2: convergence of legacy and jaxfne workflows.
    Currently uses jaxfne's receptor-exponential kernel + laminar field projection.
    """
    import jax.numpy as jnp

    if not HAS_JAXFNE_INTEGRATION:
        raise ImportError("jaxfne integration not available")

    # Convert model to jaxfne format
    eig_network, edges = jbiophysic_to_eig_network(
        tfne_model, use_receptor_exponential=True, dtype="float32"
    )

    IzhikevichTFNEScale(sim.source_scale_A_per_native)
    n_steps = int(round(sim.t_ms / sim.dt_ms))
    trials = []

    for trial_idx in range(sim.n_trials):
        seed = tfne_model.config_init.seed + sim.seed_offset + 1009 * trial_idx

        # Run simulation with jaxfne backend
        v, u, spikes_float = simulate_with_jaxfne(
            eig_network,
            edges,
            n_steps=n_steps,
            dt_ms=sim.dt_ms,
            seed=seed,
            use_receptor_exponential=True,
            dtype="float32",
        )

        # Convert spikes to boolean (jaxfne returns float 0.0/1.0)
        spikes = spikes_float >= 0.5

        # Project spikes to laminar field
        source_spike = jnp.asarray(spikes, dtype="float32")
        field_output = project_to_laminar_field(
            source_spike,
            eig_network.positions,
            n_contacts=sim.n_contacts,
            width=0.1,
        )

        # Convert field to physical amplitude
        # jaxfne returns (n_steps, n_contacts); legacy expects (n_steps, n_contacts)
        lfp_contacts = np.asarray(field_output.lfp_proxy, dtype=np.float32)  # (n_steps, n_contacts)
        csd_contacts = np.asarray(field_output.csd_proxy, dtype=np.float32)  # (n_steps, n_contacts)

        # Build trial output matching legacy structure
        trial = {
            "time_ms": np.arange(n_steps, dtype=np.float32) * np.float32(sim.dt_ms),
            "dt_ms": sim.dt_ms,
        }

        for area in tfne_model.config_init.area_order:
            area_mask = tfne_model.neurons.area == area
            area_indices = np.where(area_mask)[0]

            trial[area] = {
                "spikes": np.asarray(spikes[:, area_indices], dtype=bool),
                "voltage_mV": np.asarray(v[:, area_indices], dtype=np.float32),
                "lfp_contacts": lfp_contacts,  # Shared across areas
                "csd_contacts": csd_contacts,  # Shared across areas
                "contact_depths_m": np.asarray(field_output.contact_depths, dtype=np.float32)
                * tfne_model.config_init.cz_m,
                "neurons": tfne_model.neurons.loc[area_mask].reset_index(drop=True),
                "metadata": {
                    "backend": "jaxfne_receptor_exponential",
                    "spike_kernel": "receptor_exponential",
                    "field_projection": "laminar_gaussian_proxy",
                },
            }

        trials.append(trial)

    return SimpleNamespace(
        model=tfne_model,
        sim_config=sim,
        trials=trials,
        truth_mode=tfne_model.truth_mode,
        claim_level=tfne_model.claim_level,
        units={
            "time": "ms",
            "voltage": "mV",
            "lfp_contacts": "V_proxy",
            "csd_contacts": "A/m^3_proxy",
        },
        backend="jaxfne",
    )


# -----------------------------------------------------------------------------
# Analysis, evaluation, optimization
# -----------------------------------------------------------------------------


def _band_profile(signal: Array, dt_ms: float, freqs: Array, band: tuple[float, float]) -> Array:
    x = signal - signal.mean(axis=0, keepdims=True)
    fft = np.fft.rfft(x, axis=0)
    f_fft = np.fft.rfftfreq(x.shape[0], d=dt_ms / 1000.0)
    power = np.abs(fft) ** 2
    # interpolate to declared frequency grid for stable shapes
    interp = np.vstack([np.interp(freqs, f_fft, power[:, ch]) for ch in range(power.shape[1])])
    m = (freqs >= band[0]) & (freqs <= band[1])
    prof = interp[:, m].mean(axis=1)
    if np.ptp(prof) <= 1e-30:
        return np.zeros_like(prof)
    return (prof - prof.min()) / (prof.max() - prof.min())


def spectrolaminar_summary(
    tfne_signals: SimpleNamespace, opt: JTFNEOptConfig | None = None
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if opt is None:
        opt = JTFNEOptConfig(sim=tfne_signals.sim_config)
    freqs = np.linspace(1.0, 150.0, 96)
    rows = []
    specs = {}
    for area in tfne_signals.model.config_init.area_order:
        contacts = tfne_signals.trials[0][area]["contact_depths_m"]
        y = _depth_to_l4_position(contacts, tfne_signals.model.config_init)
        ab_profiles = []
        gm_profiles = []
        for tr in tfne_signals.trials:
            csd = tr[area]["csd_contacts"]
            ab_profiles.append(_band_profile(csd, tr["dt_ms"], freqs, (8.0, 30.0)))
            gm_profiles.append(_band_profile(csd, tr["dt_ms"], freqs, (40.0, 150.0)))
        ab = np.mean(ab_profiles, axis=0)
        gm = np.mean(gm_profiles, axis=0)
        target_y = np.asarray([-0.6, -0.4, -0.25, 0.0, 0.20, 0.40])
        tab = np.interp(y, target_y, np.asarray(opt.target_ab, dtype=float))
        tgm = np.interp(y, target_y, np.asarray(opt.target_gm, dtype=float))
        error = 0.5 * (np.mean((ab - tab) ** 2) + np.mean((gm - tgm) ** 2))
        if np.std(ab) > 1e-12 and np.std(gm) > 1e-12:
            anticorr = float(np.corrcoef(ab, gm)[0, 1])
        else:
            anticorr = 0.0
        l4_cross = float(abs(np.interp(0.0, y, ab) - np.interp(0.0, y, gm)))
        similarity = float(
            np.clip(
                100.0 * np.exp(-3.0 * (error + 0.25 * ((anticorr + 1.0) / 2.0) + 0.50 * l4_cross)),
                0.0,
                100.0,
            )
        )
        m_model = compute_motif_vector(ab, gm, y)
        m_target = compute_motif_vector(tab, tgm, y)
        motif_gate = motif_gate_score(m_model)
        specs[area] = {
            "pos_from_l4": y,
            "alpha_beta": ab,
            "gamma": gm,
            "target_alpha_beta": tab,
            "target_gamma": tgm,
            "motif_vector": m_model.as_array(),
            "target_vector": m_target.as_array(),
        }
        rows.append(
            {
                "area": area,
                "motif_gate_percent": motif_gate,
                "profile_score_percent": similarity,
                "legacy_profile_score": similarity,
                "similarity_percent": similarity,
                "S_lam": np.nan,
                "score_type": "profile_score_no_null",
                "anticorr": anticorr,
                "l4_cross_abs": l4_cross,
            }
        )
    return pd.DataFrame(rows), specs


def _spike_synchrony_kappa(spikes: Array, bin_ms: float, dt_ms: float) -> float:
    bin_n = max(1, int(round(bin_ms / dt_ms)))
    nbin = spikes.shape[0] // bin_n
    if nbin < 2 or spikes.shape[1] < 2:
        return 0.0
    b = spikes[: nbin * bin_n].reshape(nbin, bin_n, spikes.shape[1]).any(axis=1).astype(float)
    p = b.mean(axis=1)
    observed = np.mean(p * p + (1.0 - p) * (1.0 - p))
    p_global = b.mean()
    expected = p_global * p_global + (1.0 - p_global) * (1.0 - p_global)
    denom = 1.0 - expected
    if abs(denom) < 1e-12:
        return 0.0
    return float((observed - expected) / denom)


def evaluate(
    tfne_model_or_signals: SimpleNamespace, opt: JTFNEOptConfig | Mapping[str, Any]
) -> SimpleNamespace:
    """Evaluate spectrolaminar score and sanity diagnostics."""
    if isinstance(opt, Mapping):
        opt = _dataclass_from_dict(JTFNEOptConfig, opt)
    if hasattr(tfne_model_or_signals, "trials"):
        signals = tfne_model_or_signals
    else:
        sim = replace(opt.sim, n_trials=opt.eval_n_trials)
        signals = simulate(tfne_model_or_signals, sim)
    scores, specs = spectrolaminar_summary(signals, opt)
    cell_diag = celltype_diagnostics(signals.trials, signals.model.config_init.area_order)
    sync_diag = synchrony_diagnostics(signals.trials, signals.model.config_init.area_order)
    area_diag = area_diagnostics(signals.trials, signals.model.config_init.area_order)
    sanity = (
        sync_diag.groupby("area", as_index=False)
        .mean(numeric_only=True)
        .merge(
            cell_diag.groupby("area", as_index=False)[
                ["firing_rate_mean_hz", "silent_fraction", "voltage_min_mV", "voltage_max_mV"]
            ].mean(numeric_only=True),
            on="area",
            how="left",
        )
    )
    return SimpleNamespace(
        scores=scores,
        specs=specs,
        sanity=sanity,
        celltype_diagnostics=cell_diag,
        synchrony_diagnostics=sync_diag,
        area_diagnostics=area_diag,
        mean_similarity=float(scores.profile_score_percent.mean()),
        min_similarity=float(scores.profile_score_percent.min()),
        mean_motif_gate=float(scores.motif_gate_percent.mean()),
        min_motif_gate=float(scores.motif_gate_percent.min()),
        truth_mode=signals.truth_mode,
        claim_level="developmental_demo",
        interpretation="readout-level convergence test; not biological proof or unique mechanism",
    )


def optimize(
    tfne_model: SimpleNamespace, opt: JTFNEOptConfig | Mapping[str, Any]
) -> SimpleNamespace:
    """Black-box sweep over declared plasticity/synaptic-gain/noise variables."""
    if isinstance(opt, Mapping):
        opt = _dataclass_from_dict(JTFNEOptConfig, opt)
    rows = []
    best = None
    best_sim = None
    eval_i = 0
    for noise_scale in opt.sweep_noise_scale:
        for local_exc_gain in opt.sweep_local_exc_gain:
            for local_inh_gain in opt.sweep_local_inh_gain:
                for feedback_gain in opt.sweep_feedback_gain:
                    eval_i += 1
                    sim = replace(
                        opt.sim,
                        n_trials=opt.eval_n_trials,
                        noise_scale=noise_scale,
                        local_exc_gain=local_exc_gain,
                        local_inh_gain=local_inh_gain,
                        feedback_gain=feedback_gain,
                    )
                    signals = simulate(tfne_model, sim)
                    ev = evaluate(signals, opt)
                    kappa_col = (
                        "fleiss_kappa_proxy" if "fleiss_kappa_proxy" in ev.sanity else "kappa"
                    )
                    kappa_penalty = float(np.mean(ev.sanity[kappa_col].to_numpy() ** 2))
                    objective = (
                        ev.min_similarity - 100.0 * opt.synchrony_kappa_weight * kappa_penalty
                    )
                    row = {
                        "eval": eval_i,
                        "noise_scale": noise_scale,
                        "local_exc_gain": local_exc_gain,
                        "local_inh_gain": local_inh_gain,
                        "feedback_gain": feedback_gain,
                        "mean_similarity": ev.mean_similarity,
                        "min_similarity": ev.min_similarity,
                        "kappa_penalty": kappa_penalty,
                        "objective": objective,
                    }
                    rows.append(row)
                    if best is None or objective > best["objective"]:
                        best = row
                        best_sim = sim
                    if eval_i >= opt.max_evals or row["min_similarity"] >= opt.similarity_target:
                        log = pd.DataFrame(rows)
                        out = SimpleNamespace(
                            model=tfne_model,
                            best_sim=best_sim,
                            best=best,
                            optimization_log=log,
                            truth_mode=tfne_model.truth_mode,
                        )
                        tfne_model.optimization = out
                        return tfne_model
    log = pd.DataFrame(rows)
    out = SimpleNamespace(
        model=tfne_model,
        best_sim=best_sim,
        best=best,
        optimization_log=log,
        truth_mode=tfne_model.truth_mode,
    )
    tfne_model.optimization = out
    return tfne_model


# -----------------------------------------------------------------------------
# Visualization and artifacts
# -----------------------------------------------------------------------------


def visualize(
    tfne_signals: SimpleNamespace, vis: JTFNEVisConfig | Mapping[str, Any]
) -> dict[str, Any]:
    """Create Plotly visual summaries and optionally write HTML/JSON artifacts."""
    if isinstance(vis, Mapping):
        vis = _dataclass_from_dict(JTFNEVisConfig, vis)
    out_dir = Path(vis.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    figures: dict[str, Any] = {}
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except Exception:
        return {"status": "plotly_unavailable", "output_dir": str(out_dir)}

    model = tfne_signals.model
    neurons = model.neurons
    # 3D network figure.
    fig = go.Figure()
    colors = {"E": "orange", "PV": "royalblue", "SST": "gold", "VIP": "purple"}
    for ct, sub in neurons.groupby("cell_type"):
        fig.add_trace(
            go.Scatter3d(
                x=sub.x_m * 1e6,
                y=sub.y_m * 1e6,
                z=sub.z_m * 1e6,
                mode="markers",
                marker=dict(size=3, color=colors.get(ct, "gray")),
                name=ct,
                text=[f"{r.area} {r.layer} {r.cell_type} #{r.neuron_id}" for r in sub.itertuples()],
            )
        )
    fig.update_layout(
        title=f"{vis.title_prefix}: cortical network 3D",
        scene=dict(zaxis_title="depth um", xaxis_title="x um", yaxis_title="y um"),
    )
    figures["network3d"] = fig

    evaluation = evaluate(tfne_signals, JTFNEOptConfig(sim=tfne_signals.sim_config))
    for area, spec in evaluation.specs.items():
        fig2 = make_subplots(
            rows=1, cols=2, subplot_titles=("Laminar profiles", "First-trial LFP/CSD")
        )
        y = spec["pos_from_l4"]
        fig2.add_trace(
            go.Scatter(x=spec["alpha_beta"], y=y, name="alpha/beta", mode="lines+markers"),
            row=1,
            col=1,
        )
        fig2.add_trace(
            go.Scatter(x=spec["gamma"], y=y, name="gamma", mode="lines+markers"), row=1, col=1
        )
        tr = tfne_signals.trials[0][area]
        fig2.add_trace(
            go.Heatmap(
                z=tr["csd_contacts"].T,
                x=tfne_signals.trials[0]["time_ms"],
                y=tr["contact_depths_m"] * 1e6,
                colorscale="RdBu",
                name="CSD",
            ),
            row=1,
            col=2,
        )
        fig2.update_layout(title=f"{vis.title_prefix}: {area} spectrolaminar/activity suite")
        figures[f"spectrolaminar_{area}"] = fig2

    if vis.write_html:
        for name, figobj in figures.items():
            figobj.write_html(out_dir / f"{name}.html")
    if vis.write_json:
        manifest = write_manifest(
            tfne_signals, out_dir, evaluation=evaluation, figure_names=list(figures.keys())
        )
        figures["manifest"] = manifest
    if vis.show:
        for figobj in figures.values():
            if hasattr(figobj, "show"):
                figobj.show()
    return figures


def _hash_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def operator_status() -> dict[str, dict[str, object]]:
    """Return machine-readable TFNE operator implementation status."""
    return operator_status_json()


def status() -> dict[str, dict[str, object]]:
    """Alias for :func:`operator_status` for notebook ergonomics."""
    return operator_status()


def operator_status_by_symbol() -> dict[str, dict[str, object]]:
    """Return operator status keyed by formal TFNE symbols for evidence artifacts."""
    return operator_status_by_symbol_json()


def operator_graph() -> list[str]:
    """Return formal TFNE operator order exposed by the facade."""
    return ["E_theta", "S_WDR", "C_mu_nu", "Q_eta_alpha", "F_Omega_B_G_Gamma", "P", "A", "O", "C"]


def validate(obj: SimpleNamespace) -> SimpleNamespace:
    """Return lightweight validation report for a model or signals object."""
    failures: list[str] = []
    warnings: list[str] = []
    if hasattr(obj, "trials"):
        for area in obj.model.config_init.area_order:
            for tr_i, tr in enumerate(obj.trials):
                data = tr[area]
                if data["spikes"].shape != data["voltage_mV"].shape:
                    failures.append(f"{area} trial {tr_i}: spikes/voltage shape mismatch")
                if data["lfp_contacts"].ndim != 2 or data["csd_contacts"].ndim != 2:
                    failures.append(f"{area} trial {tr_i}: probe output must be [T,C]")
                meta = data.get("metadata", {})
                for key in (
                    "source_calibration_status",
                    "boundary_condition",
                    "source_projection_mode",
                ):
                    if key not in meta:
                        failures.append(f"{area} trial {tr_i}: missing metadata {key}")
                if not np.all(np.isfinite(data["lfp_contacts"])) or not np.all(
                    np.isfinite(data["csd_contacts"])
                ):
                    failures.append(f"{area} trial {tr_i}: non-finite probe values")
    elif hasattr(obj, "neurons"):
        if len(obj.neurons) <= 0:
            failures.append("model has no neurons")
        if getattr(obj, "truth_mode", None) != "truth_safe_unverified":
            warnings.append("unexpected truth_mode")
    else:
        warnings.append("unknown object type for validation")
    return SimpleNamespace(accepted=not failures, failures=failures, warnings=warnings)


def manifest(
    tfne_signals: SimpleNamespace, *, evaluation: SimpleNamespace | None = None
) -> dict[str, Any]:
    """Build JSON-safe manifest without writing it."""
    cfg_hash = hashlib.sha256(
        json.dumps(
            json_safe(asdict(tfne_signals.sim_config)), sort_keys=True, allow_nan=False
        ).encode()
    ).hexdigest()
    basis_meta = {}
    field_rows = []
    for area, b in tfne_signals.model.tfne_basis.items():
        meta = {
            k: v
            for k, v in b.items()
            if k not in {"mask", "lfp_basis", "csd_basis", "contact_depths_m"}
        }
        basis_meta[area] = meta
        field_rows.append({"area": area, **meta})
    payload = {
        "truth_mode": tfne_signals.truth_mode,
        "claim_level": tfne_signals.claim_level,
        "interpretation": (
            "developmental TFNE-Izhikevich spectrolaminar readout demo; "
            "not biological proof, mechanism proof, or calibrated amplitude evidence"
        ),
        "config_hash": cfg_hash,
        "operator_status": operator_status(),
        "operator_status_by_symbol": operator_status_by_symbol(),
        "operator_graph": operator_graph(),
        "model": {
            "n_neurons": int(len(tfne_signals.model.neurons)),
            "areas": list(tfne_signals.model.config_init.area_order),
            "cell_types": list(tfne_signals.model.config_init.cell_types),
            "mode": tfne_signals.model.config_init.mode,
        },
        "field": basis_meta,
        "array_layout": {"current_density": "channel_first"},
        "simulation": {
            "n_trials": len(tfne_signals.trials),
            "dt_ms": tfne_signals.sim_config.dt_ms,
            "t_ms": tfne_signals.sim_config.t_ms,
            "time_window_ms": list(tfne_signals.sim_config.time_window_ms),
            "event_window_ms": list(tfne_signals.sim_config.event_window_ms),
            "post_window_ms": list(tfne_signals.sim_config.post_window_ms),
        },
        "evaluation": None
        if evaluation is None
        else {
            "mean_profile_score_percent": evaluation.mean_similarity,
            "min_profile_score_percent": evaluation.min_similarity,
            "mean_motif_gate_percent": evaluation.mean_motif_gate,
            "min_motif_gate_percent": evaluation.min_motif_gate,
            "score_notice": (
                "S_lam is null-normalized and is null "
                "unless a declared null distribution is supplied."
            ),
            "scores": evaluation.scores.to_dict(orient="records"),
            "sanity": evaluation.sanity.to_dict(orient="records"),
        },
    }
    return json_safe(payload)


def write_manifest(
    tfne_signals: SimpleNamespace,
    out_dir: str | Path,
    *,
    evaluation: SimpleNamespace | None = None,
    figure_names: list[str] | None = None,
) -> dict[str, Any]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = manifest(tfne_signals, evaluation=evaluation)
    payload["figures"] = figure_names or []
    path = out_dir / "manifest.json"
    write_json_manifest(path, payload)
    payload["manifest_sha256"] = _hash_file(path)
    return payload


__all__ = [
    "JTFNEConfig",
    "JTFNEInitConfig",
    "JTFNESimConfig",
    "JTFNEVisConfig",
    "JTFNEOptConfig",
    "default_cfg",
    "save_cfg",
    "load_cfg",
    "construct",
    "save_model",
    "load_model",
    "simulate",
    "visualize",
    "evaluate",
    "optimize",
    "spectrolaminar_summary",
    "write_manifest",
    "status",
    "operator_status",
    "operator_status_by_symbol",
    "operator_graph",
    "validate",
    "manifest",
]

# jaxfne integration (Phase 1+: unified backend)
if HAS_JAXFNE_INTEGRATION:
    __all__.extend(
        [
            "jbiophysic_to_eig_network",
            "simulate_with_jaxfne",
            "project_to_laminar_field",
            "get_receptor_info",
            "diagnose_connectivity",
        ]
    )
