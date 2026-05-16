"""Izhikevich implementation of the E/Ing/Inl network motif.

This module mirrors the topology of the JAXley ``net_eig`` scaffold that used
HH cells, graded AMPA synapses, and graded GABAa synapses.  The replacement is a
lightweight point-neuron network with explicit Izhikevich emitter parameters and
edge-list synapse metadata.

Population semantics
--------------------
- ``E``: excitatory regular-spiking cells.
- ``Ing``: global inhibitory interneurons, SST-like / low-threshold-spiking.
- ``Inl``: local inhibitory interneurons, PV-like / fast-spiking.

Connectivity semantics match the source code supplied by the project:
- E -> all, AMPA, tauD = 2 ms.
- Ing -> all, GABAa, tauD = 5 ms.
- Inl -> selected 10% posts, GABAa, tauD = 2 ms.

The original JAXley scaffold used two branches and connected branch 0 at loc 0.0
to branch 1 at loc 1.0.  Because Izhikevich emitters are point-neuron dynamics,
this module preserves those branch/location values as connection-site metadata
rather than pretending to have multicompartment morphology.

Native Izhikevich current is a current-like model drive, not physical amperes.
Use the TFNE calibration bridge before making physical source/CSD/LFP amplitude
claims.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from jbiophysic.cells.izhikevich import IzhikevichParams


@dataclass(frozen=True)
class EIGSynapseSpec:
    """Static edge-list representation for one connection family."""

    pre: np.ndarray
    post: np.ndarray
    receptor: str
    tauD_ms: float
    sign: float
    weight: float
    pre_branch: int = 0
    pre_loc: float = 0.0
    post_branch: int = 1
    post_loc: float = 1.0
    synapse_model: str = "graded"

    def to_edge_records(self) -> list[dict[str, Any]]:
        """Return JSON-safe edge records."""
        return [
            {
                "pre": int(pre),
                "post": int(post),
                "receptor": self.receptor,
                "tauD_ms": float(self.tauD_ms),
                "sign": float(self.sign),
                "weight": float(self.weight),
                "pre_branch": int(self.pre_branch),
                "pre_loc": float(self.pre_loc),
                "post_branch": int(self.post_branch),
                "post_loc": float(self.post_loc),
                "synapse_model": self.synapse_model,
            }
            for pre, post in zip(self.pre, self.post, strict=True)
        ]


@dataclass(frozen=True)
class IzhikevichEIGNetwork:
    """A deterministic Izhikevich E/Ing/Inl network specification.

    This is a pure specification plus a smoke-scale simulator.  It is intentionally
    separate from JAXley because the cells are reduced point emitters, not HH
    compartments.
    """

    num_e: int
    num_ig: int
    num_il: int
    seed: int
    local_connection_key: int
    cell_type: np.ndarray
    group_index: Mapping[str, np.ndarray]
    izhikevich: Mapping[str, np.ndarray]
    noise_tau_ms: np.ndarray
    noise_amp: np.ndarray
    radius: np.ndarray
    length: np.ndarray
    synapses: tuple[EIGSynapseSpec, ...]
    source_calibration_status: str = "uncalibrated_izhikevich_native_current"
    source_claim: str = "reduced_emitter_spike_state_only_not_physical_current"
    branch_semantics: Mapping[str, Any] = field(
        default_factory=lambda: {
            "n_logical_branches": 2,
            "pre_branch": 0,
            "pre_loc": 0.0,
            "post_branch": 1,
            "post_loc": 1.0,
            "note": "Preserved from JAXley scaffold as metadata for point-neuron emitters.",
        }
    )

    @property
    def n_neurons(self) -> int:
        return int(self.num_e + self.num_ig + self.num_il)

    @property
    def edge_count(self) -> int:
        return int(sum(spec.pre.size for spec in self.synapses))

    @property
    def edge_arrays(self) -> dict[str, np.ndarray]:
        """Return flattened edge arrays suitable for simulation or export."""
        if not self.synapses:
            return {
                "pre": np.asarray([], dtype=np.int32),
                "post": np.asarray([], dtype=np.int32),
                "sign": np.asarray([], dtype=np.float32),
                "weight": np.asarray([], dtype=np.float32),
                "tauD_ms": np.asarray([], dtype=np.float32),
                "receptor_index": np.asarray([], dtype=np.int8),
            }
        receptors = {"AMPA": 0, "GABAa": 1}
        pre = np.concatenate([s.pre for s in self.synapses]).astype(np.int32)
        post = np.concatenate([s.post for s in self.synapses]).astype(np.int32)
        sign = np.concatenate([np.full(s.pre.size, s.sign) for s in self.synapses]).astype(
            np.float32
        )
        weight = np.concatenate([np.full(s.pre.size, s.weight) for s in self.synapses]).astype(
            np.float32
        )
        tau = np.concatenate([np.full(s.pre.size, s.tauD_ms) for s in self.synapses]).astype(
            np.float32
        )
        receptor_index = np.concatenate(
            [np.full(s.pre.size, receptors[s.receptor]) for s in self.synapses]
        ).astype(np.int8)
        return {
            "pre": pre,
            "post": post,
            "sign": sign,
            "weight": weight,
            "tauD_ms": tau,
            "receptor_index": receptor_index,
        }

    def counts_by_group(self) -> dict[str, int]:
        return {name: int(indices.size) for name, indices in self.group_index.items()}

    def connection_summary(self) -> dict[str, int]:
        return {f"{spec.receptor}_{spec.tauD_ms:g}ms": int(spec.pre.size) for spec in self.synapses}

    def to_json_dict(self) -> dict[str, Any]:
        """Return a JSON-safe dictionary for manifests."""
        return {
            "model": "izhikevich_eig_network",
            "num_e": int(self.num_e),
            "num_ig": int(self.num_ig),
            "num_il": int(self.num_il),
            "n_neurons": int(self.n_neurons),
            "seed": int(self.seed),
            "local_connection_key": int(self.local_connection_key),
            "cell_type": self.cell_type.tolist(),
            "groups": {
                name: indices.astype(int).tolist() for name, indices in self.group_index.items()
            },
            "izhikevich": {k: np.asarray(v).tolist() for k, v in self.izhikevich.items()},
            "noise_tau_ms": self.noise_tau_ms.tolist(),
            "noise_amp": self.noise_amp.tolist(),
            "radius": self.radius.tolist(),
            "length": self.length.tolist(),
            "source_calibration_status": self.source_calibration_status,
            "source_claim": self.source_claim,
            "branch_semantics": dict(self.branch_semantics),
            "edge_count": int(self.edge_count),
            "connection_summary": self.connection_summary(),
            "synapses": [record for spec in self.synapses for record in spec.to_edge_records()],
        }


def _validate_counts(num_e: int, num_ig: int, num_il: int) -> None:
    for name, value in {"num_e": num_e, "num_ig": num_ig, "num_il": num_il}.items():
        if int(value) != value or value < 0:
            raise ValueError(f"{name} must be a nonnegative integer")
    if num_e + num_ig + num_il <= 0:
        raise ValueError("network must contain at least one neuron")


def _fully_connect(pre: np.ndarray, post: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return all ordered pre/post edge pairs, preserving self-edges across logical branches."""
    pre = np.asarray(pre, dtype=np.int32)
    post = np.asarray(post, dtype=np.int32)
    if pre.size == 0 or post.size == 0:
        return np.asarray([], dtype=np.int32), np.asarray([], dtype=np.int32)
    pre_grid = np.repeat(pre, post.size)
    post_grid = np.tile(post, pre.size)
    return pre_grid.astype(np.int32), post_grid.astype(np.int32)


def _selected_local_posts(n_total: int, *, key_seed: int, local_fraction: float) -> np.ndarray:
    if not 0.0 <= local_fraction <= 1.0:
        raise ValueError("local_fraction must be in [0, 1]")
    n_select = int(n_total * local_fraction)
    if n_select <= 0:
        return np.asarray([], dtype=np.int32)
    posts_pool = jnp.arange(0, n_total)
    key = jax.random.PRNGKey(key_seed)
    selected = jax.random.choice(key, posts_pool, shape=(n_select,), replace=False)
    return np.asarray(selected, dtype=np.int32)


def net_eig(
    num_e: int,
    num_ig: int,
    num_il: int,
    *,
    seed: int = 0,
    local_connection_key: int = 1,
    local_fraction: float = 0.10,
    e_weight: float = 0.05,
    ig_weight: float = 0.06,
    il_weight: float = 0.08,
) -> IzhikevichEIGNetwork:
    """Construct the JAXley ``net_eig`` motif using Izhikevich point neurons.

    Parameters mirror the supplied JAXley scaffold: ``num_e`` excitatory cells,
    ``num_ig`` global inhibitory/SST-like cells, and ``num_il`` local
    inhibitory/PV-like cells.  The topology, receptor classes, synaptic decay
    constants, group labels, two logical connection sites, and 10% local-post
    selection are preserved.

    Returns
    -------
    IzhikevichEIGNetwork
        A deterministic network specification and smoke-scale simulator input.
    """

    _validate_counts(num_e, num_ig, num_il)
    n_total = int(num_e + num_ig + num_il)
    rng = np.random.default_rng(seed)

    e_idx = np.arange(0, num_e, dtype=np.int32)
    ig_idx = np.arange(num_e, num_e + num_ig, dtype=np.int32)
    il_idx = np.arange(num_e + num_ig, n_total, dtype=np.int32)
    all_idx = np.arange(0, n_total, dtype=np.int32)

    cell_type = np.empty(n_total, dtype=object)
    cell_type[e_idx] = "E"
    cell_type[ig_idx] = "Ing"
    cell_type[il_idx] = "Inl"

    # Preserve the original Inoise initialization logic, including the clipped
    # nonnegative amplitude range.  These are native current-like noise parameters.
    noise_tau_ms = rng.uniform(5.0, 50.0, size=n_total).astype(np.float32)
    noise_amp = np.clip(rng.uniform(-0.1, 0.1, size=n_total), 0.0, 0.1).astype(np.float32)

    # E: regular spiking; Ing/SST-like: low-threshold spiking; Inl/PV-like: fast spiking.
    a = np.empty(n_total, dtype=np.float32)
    b = np.empty(n_total, dtype=np.float32)
    c = np.empty(n_total, dtype=np.float32)
    d = np.empty(n_total, dtype=np.float32)
    v_spike = np.full(n_total, 30.0, dtype=np.float32)
    a[e_idx], b[e_idx], c[e_idx], d[e_idx] = 0.02, 0.20, -65.0, 8.0
    a[ig_idx], b[ig_idx], c[ig_idx], d[ig_idx] = 0.02, 0.25, -65.0, 2.0
    a[il_idx], b[il_idx], c[il_idx], d[il_idx] = 0.10, 0.20, -65.0, 2.0

    pre_e, post_all_for_e = _fully_connect(e_idx, all_idx)
    pre_ig, post_all_for_ig = _fully_connect(ig_idx, all_idx)
    selected_posts = _selected_local_posts(
        n_total, key_seed=local_connection_key, local_fraction=local_fraction
    )
    pre_il, post_il = _fully_connect(il_idx, selected_posts)

    synapses = (
        EIGSynapseSpec(
            pre=pre_e,
            post=post_all_for_e,
            receptor="AMPA",
            tauD_ms=2.0,
            sign=1.0,
            weight=float(e_weight),
            synapse_model="graded_ampa",
        ),
        EIGSynapseSpec(
            pre=pre_ig,
            post=post_all_for_ig,
            receptor="GABAa",
            tauD_ms=5.0,
            sign=-1.0,
            weight=float(ig_weight),
            synapse_model="graded_gabaa",
        ),
        EIGSynapseSpec(
            pre=pre_il,
            post=post_il,
            receptor="GABAa",
            tauD_ms=2.0,
            sign=-1.0,
            weight=float(il_weight),
            synapse_model="graded_gabaa",
        ),
    )

    return IzhikevichEIGNetwork(
        num_e=int(num_e),
        num_ig=int(num_ig),
        num_il=int(num_il),
        seed=int(seed),
        local_connection_key=int(local_connection_key),
        cell_type=cell_type,
        group_index={"E": e_idx, "Ing": ig_idx, "Inl": il_idx},
        izhikevich={"a": a, "b": b, "c": c, "d": d, "v_spike_mV": v_spike},
        noise_tau_ms=noise_tau_ms,
        noise_amp=noise_amp,
        radius=np.ones(n_total, dtype=np.float32),
        length=np.ones(n_total, dtype=np.float32),
        synapses=synapses,
    )


def make_izhikevich_eig_network(*args: Any, **kwargs: Any) -> IzhikevichEIGNetwork:
    """Descriptive alias for :func:`net_eig`."""
    return net_eig(*args, **kwargs)


def simulate_eig_izhikevich(
    network: IzhikevichEIGNetwork,
    drive: np.ndarray,
    *,
    dt_ms: float = 0.5,
    seed: int | None = None,
    noise_scale: float = 1.0,
    v0_mV: float = -65.0,
) -> dict[str, np.ndarray]:
    """Run a smoke-scale Euler simulation of an :class:`IzhikevichEIGNetwork`.

    ``drive`` must have shape ``[time, neuron]`` in native Izhikevich current-like
    units.  Returned voltages, spikes, synaptic currents, and final synaptic
    states are intended for smoke tests and objective prototyping, not calibrated
    physical amplitude claims.
    """

    if dt_ms <= 0.0:
        raise ValueError("dt_ms must be positive")
    drive = np.asarray(drive, dtype=np.float32)
    if drive.ndim != 2 or drive.shape[1] != network.n_neurons:
        raise ValueError("drive must have shape [time, network.n_neurons]")
    if not np.all(np.isfinite(drive)):
        raise ValueError("drive contains non-finite values")

    sim_seed = network.seed if seed is None else int(seed)
    rng = np.random.default_rng(sim_seed)
    steps, n = drive.shape
    params = network.izhikevich
    a = np.asarray(params["a"], dtype=np.float32)
    b = np.asarray(params["b"], dtype=np.float32)
    c = np.asarray(params["c"], dtype=np.float32)
    d = np.asarray(params["d"], dtype=np.float32)
    v_spike = np.asarray(params["v_spike_mV"], dtype=np.float32)

    edges = network.edge_arrays
    pre = edges["pre"]
    post = edges["post"]
    sign = edges["sign"]
    weight = edges["weight"]
    tau = edges["tauD_ms"]
    decay = np.exp(-dt_ms / np.maximum(tau, 1.0e-6)).astype(np.float32)

    v = (v0_mV + 2.0 * rng.standard_normal(n)).astype(np.float32)
    u = (b * v).astype(np.float32)
    syn_state = np.zeros(pre.shape[0], dtype=np.float32)

    V = np.zeros((steps, n), dtype=np.float32)
    U = np.zeros((steps, n), dtype=np.float32)
    spikes = np.zeros((steps, n), dtype=bool)
    I_syn_trace = np.zeros((steps, n), dtype=np.float32)

    for t in range(steps):
        I_syn = np.zeros(n, dtype=np.float32)
        if pre.size:
            np.add.at(I_syn, post, sign * weight * syn_state)
        noise = noise_scale * network.noise_amp * rng.standard_normal(n).astype(np.float32)
        current_in = drive[t] + I_syn + noise

        dv = 0.04 * v * v + 5.0 * v + 140.0 - u + current_in
        du = a * (b * v - u)
        v_pre = v + dt_ms * dv
        u_pre = u + dt_ms * du
        spiked = v_pre >= v_spike
        v = np.where(spiked, c, v_pre).astype(np.float32)
        u = np.where(spiked, u_pre + d, u_pre).astype(np.float32)

        if pre.size:
            syn_state = syn_state * decay + spiked[pre].astype(np.float32)

        V[t] = v
        U[t] = u
        spikes[t] = spiked
        I_syn_trace[t] = I_syn

    duration_s = steps * dt_ms / 1000.0
    firing_rate_hz = spikes.sum(axis=0).astype(np.float32) / max(duration_s, 1.0e-12)
    return {
        "V_mV": V,
        "U": U,
        "spikes": spikes,
        "I_syn_native": I_syn_trace,
        "firing_rate_hz": firing_rate_hz,
        "final_syn_state": syn_state,
        "finite": np.asarray(np.isfinite(V).all() and np.isfinite(I_syn_trace).all()),
        "source_calibration_status": np.asarray(network.source_calibration_status, dtype=object),
        "claim_status": np.asarray(network.source_claim, dtype=object),
    }


__all__ = [
    "EIGSynapseSpec",
    "IzhikevichEIGNetwork",
    "make_izhikevich_eig_network",
    "net_eig",
    "simulate_eig_izhikevich",
    "IzhikevichParams",
]
