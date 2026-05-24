"""Integration layer: jbiophysic → jaxfne unified backend.

This module provides conversion utilities to map jbiophysic's neuron and network
models to jaxfne's architecture. It enables using jaxfne's receptor-indexed
exponential synaptic kernel and laminar forward-field projection in place of
jbiophysic's custom implementations.

Architecture mapping:
- jbiophysic neurons (per-neuron a,b,c,d in DataFrame) → jaxfne IzhikevichParams (population-level arrays)
- jbiophysic connectivity (W_local_exc, W_local_inh, feedforward, feedback) → jaxfne EdgeList with receptor indices
- jbiophysic TFNE forward-field solver → jaxfne.project_laminar_sources (Gaussian laminar proxy)
"""

from __future__ import annotations

from typing import Any, Mapping, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd

# Guarded jaxfne import: required for this module, optional for the package.
try:
    import jaxfne as jtfne
    from jaxfne import (
        EdgeList,
        EIGNetwork,
        IzhikevichParams,
        ReceptorSpec,
        standard_receptor_specs,
    )
except ImportError as exc:
    raise ImportError(
        "jaxfne is required for jbiophysic.jaxfne_integration. "
        "Install it with: pip install -e '.[jaxfne]'"
    ) from exc

Array = jax.Array


class ReceptorIndexing(NamedTuple):
    """Receptor type → index mapping for jaxfne's receptor-indexed kernel."""

    AMPA: int = 0  # E→E, E→PV, E→SST, E→VIP (excitatory)
    GABA_A: int = 1  # All inhibitory (PV→*, SST→*, VIP→*)
    NMDA: int = 2  # (Optional) NMDA channels
    GABA_B: int = 3  # (Optional) GABA_B slow


def jbiophysic_to_eig_network(
    model: Any,  # SimpleNamespace from jtfne.construct()
    *,
    use_receptor_exponential: bool = True,
    dtype: str = "float32",
) -> tuple[EIGNetwork, EdgeList | None]:
    """Convert jbiophysic model to jaxfne EIGNetwork + EdgeList.

    Parameters
    ----------
    model : SimpleNamespace
        Output from jbiophysic.jtfne.construct(). Must contain:
        - neurons: DataFrame with columns [neuron_id, cell_type, layer, layer_label,
                   a, b, c, d, v_spike_mV, x_m, y_m, z_m]
        - positions_m: (n_neurons, 3) positions in meters
        - W_parts: dict with keys [local_exc, local_inh, feedforward, feedback]
        - config_init: JTFNEInitConfig (for parameters like seed, layer bounds)

    use_receptor_exponential : bool, default True
        If True, return EdgeList for use with simulate_receptor_exponential_izhikevich.
        If False, return None for EdgeList (use simulate_eig_izhikevich instead, no receptors).

    dtype : str, default 'float32'
        JAX dtype for arrays.

    Returns
    -------
    eig_network : EIGNetwork
        Lightweight EIGNetwork with IzhikevichParams and positions.
    edges : EdgeList or None
        Sparse connectivity with receptor indices (if use_receptor_exponential=True).
        None if use_receptor_exponential=False (W embedded in params.W instead).

    Notes
    -----
    - Neuron positions are converted to relative laminar depth [0,1] in the third coordinate.
    - Receptor assignment:
      - E→E, E→PV: AMPA (index 0)
      - E→SST, E→VIP: AMPA (index 0)
      - PV→all, SST→all, VIP→all: GABA_A (index 1)
    - Weight matrices use standard recurrent convention: rows = post, cols = pre.
    - Seed reproducibility is preserved via deterministic neuron ordering.
    """
    neurons = model.neurons
    positions_m = model.positions_m
    W_parts = model.W_parts
    init = model.config_init

    n_neurons = len(neurons)

    # Extract per-neuron Izhikevich parameters as arrays
    a = jnp.asarray(neurons["a"].to_numpy(), dtype=dtype)
    b = jnp.asarray(neurons["b"].to_numpy(), dtype=dtype)
    c = jnp.asarray(neurons["c"].to_numpy(), dtype=dtype)
    d = jnp.asarray(neurons["d"].to_numpy(), dtype=dtype)

    # Initialize state arrays
    v0 = jnp.full(n_neurons, -65.0, dtype=dtype)  # Standard resting potential
    u0 = b * v0  # Recovery variable at rest
    drive = jnp.zeros(n_neurons, dtype=dtype)  # Will be set during simulation

    # Cell type sign: E=+1, others=-1 (though this is metadata in jaxfne)
    cell_types = neurons["cell_type"].to_numpy()
    sign = jnp.asarray([1.0 if ct == "E" else -1.0 for ct in cell_types], dtype=dtype)

    # Source scale (amplitude factor for field projection)
    source_scale = jnp.ones(n_neurons, dtype=dtype)

    # Cell type and layer labels for metadata
    labels = tuple(cell_types)
    layer_labels = tuple(neurons["layer"].to_numpy())

    # Convert positions to relative laminar depth [0, 1]
    z_m = positions_m[:, 2]
    z_min, z_max = z_m.min(), z_m.max()
    z_depth_rel = (z_m - z_min) / (z_max - z_min + 1e-12) if z_max > z_min else jnp.ones_like(z_m) * 0.5
    positions_normalized = jnp.stack(
        [positions_m[:, 0], positions_m[:, 1], z_depth_rel],
        axis=1,
    ).astype(dtype)

    # Create IzhikevichParams (population-level container)
    if use_receptor_exponential:
        # No W in params when using EdgeList; will be specified in EdgeList
        W_empty = jnp.zeros((n_neurons, n_neurons), dtype=dtype)
    else:
        # W embedded in params for simulate_eig_izhikevich
        W_empty = jnp.asarray(
            (
                W_parts["local_exc"]
                + W_parts["local_inh"]
                + W_parts["feedforward"]
                + W_parts["feedback"]
            ),
            dtype=dtype,
        )

    params = IzhikevichParams(
        a=a,
        b=b,
        c=c,
        d=d,
        drive=drive,
        sign=sign,
        W=W_empty,
        v0=v0,
        u0=u0,
        source_scale=source_scale,
        labels=labels,
        layer_labels=layer_labels,
        source_calibration_status="uncalibrated_izhikevich_native_current",
    )

    # Create EIGNetwork
    layer_labels_arr = np.array(layer_labels)
    eig_network = EIGNetwork(
        params=params,
        positions=positions_normalized,
        metadata={
            "n_neurons": n_neurons,
            "cell_type_fractions": {
                ct: (cell_types == ct).sum() / n_neurons for ct in np.unique(cell_types)
            },
            "layer_fractions": {
                layer: (layer_labels_arr == layer).sum() / n_neurons
                for layer in np.unique(layer_labels_arr)
            },
            "area_order": tuple(init.area_order),
        },
    )

    # Build EdgeList if using receptor exponential kernel
    edges = None
    if use_receptor_exponential:
        edges = _build_edges_from_connectivity(
            neurons=neurons,
            W_parts=W_parts,
            dtype=dtype,
        )

    return eig_network, edges


def _build_edges_from_connectivity(
    neurons: pd.DataFrame,
    W_parts: dict[str, np.ndarray],
    dtype: str = "float32",
) -> EdgeList:
    """Convert jbiophysic connectivity matrices to jaxfne EdgeList.

    Parameters
    ----------
    neurons : DataFrame
        Must have columns: [neuron_id, cell_type, layer]

    W_parts : dict
        Keys: [local_exc, local_inh, feedforward, feedback]
        Values: (n, n) dense weight matrices (rows=post, cols=pre)

    dtype : str
        JAX dtype for arrays.

    Returns
    -------
    EdgeList
        Sparse connectivity with receptor indices.
    """
    cell_types = neurons["cell_type"].to_numpy()
    pre_indices = []
    post_indices = []
    weights = []
    receptor_indices = []
    tau_ms_list = []

    receptor_specs = standard_receptor_specs()
    receptors = ReceptorIndexing()

    # Process local excitatory (E→all, AMPA)
    W_local_exc = W_parts["local_exc"]
    post, pre = np.where(W_local_exc != 0)
    for p, q in zip(pre, post):
        if cell_types[p] != "E":
            continue  # Skip non-E sources
        pre_indices.append(p)
        post_indices.append(q)
        weights.append(W_local_exc[q, p])
        receptor_indices.append(receptors.AMPA)
        tau_ms_list.append(receptor_specs["AMPA"].tau_ms)

    # Process local inhibitory (I→all, GABA_A)
    W_local_inh = W_parts["local_inh"]
    post, pre = np.where(W_local_inh != 0)
    for p, q in zip(pre, post):
        if cell_types[p] == "E":
            continue  # Skip E sources
        pre_indices.append(p)
        post_indices.append(q)
        weights.append(W_local_inh[q, p])
        receptor_indices.append(receptors.GABA_A)
        tau_ms_list.append(receptor_specs["GABA_A"].tau_ms)

    # Process feedforward (E→all, AMPA)
    W_ff = W_parts["feedforward"]
    post, pre = np.where(W_ff != 0)
    for p, q in zip(pre, post):
        pre_indices.append(p)
        post_indices.append(q)
        weights.append(W_ff[q, p])
        receptor_indices.append(receptors.AMPA)
        tau_ms_list.append(receptor_specs["AMPA"].tau_ms)

    # Process feedback (E→all, AMPA)
    W_fb = W_parts["feedback"]
    post, pre = np.where(W_fb != 0)
    for p, q in zip(pre, post):
        pre_indices.append(p)
        post_indices.append(q)
        weights.append(W_fb[q, p])
        receptor_indices.append(receptors.AMPA)
        tau_ms_list.append(receptor_specs["AMPA"].tau_ms)

    # Construct JAX arrays
    pre_arr = jnp.asarray(pre_indices, dtype="int32")
    post_arr = jnp.asarray(post_indices, dtype="int32")
    weight_arr = jnp.asarray(weights, dtype=dtype)
    receptor_arr = jnp.asarray(receptor_indices, dtype="int32")
    tau_arr = jnp.asarray(tau_ms_list, dtype=dtype)

    return EdgeList(
        pre=pre_arr,
        post=post_arr,
        weight=weight_arr,
        receptor_index=receptor_arr,
        tau_ms=tau_arr,
        source_calibration_status="uncalibrated_izhikevich_native_current",
    )


def simulate_with_jaxfne(
    eig_network: EIGNetwork,
    edges: EdgeList | None,
    n_steps: int,
    dt_ms: float,
    seed: int,
    *,
    drive_schedule: Array | None = None,
    use_receptor_exponential: bool = True,
    dtype: str = "float32",
) -> tuple[Array, Array, Array]:
    """Simulate using jaxfne kernel.

    Parameters
    ----------
    eig_network : EIGNetwork
        Network built by jbiophysic_to_eig_network().

    edges : EdgeList or None
        Sparse connectivity (required if use_receptor_exponential=True).

    n_steps : int
        Number of simulation steps.

    dt_ms : float
        Integration timestep in milliseconds.

    seed : int
        PRNG seed.

    drive_schedule : Array, optional
        Shape (n_steps, n_neurons) applied current schedule.

    use_receptor_exponential : bool, default True
        Use receptor-exponential kernel or simpler EIG kernel.

    dtype : str
        JAX dtype.

    Returns
    -------
    v : (n_steps, n_neurons)
        Membrane voltage trace.

    u : (n_steps, n_neurons)
        Recovery variable trace.

    spikes : (n_steps, n_neurons)
        Boolean spike raster.
    """
    key = jax.random.key(seed)

    if use_receptor_exponential:
        if edges is None:
            raise ValueError("edges required when use_receptor_exponential=True")
        v, u, spikes, _ = jaxfne.simulate_receptor_exponential_izhikevich(
            eig_network.params,
            edges,
            n_steps=n_steps,
            dt_ms=dt_ms,
            key=key,
            dtype=dtype,
            drive_schedule=drive_schedule,
        )
    else:
        v, u, spikes = jaxfne.simulate_eig_izhikevich(
            eig_network.params,
            n_steps=n_steps,
            dt_ms=dt_ms,
            key=key,
            dtype=dtype,
            drive_schedule=drive_schedule,
        )

    return v, u, spikes


def get_receptor_info() -> dict[str, dict]:
    """Return jaxfne's standard receptor kinetics info.

    Useful for understanding receptor properties (tau, sign, reversal potential)
    used in simulations.

    Returns
    -------
    dict
        Mapping receptor names to spec dicts with keys:
        - receptor_index: int (0-3)
        - sign: int (1 for excitatory, -1 for inhibitory)
        - tau_ms: float (time constant in milliseconds)
        - reversal_mV: float or None (reversal potential)
        - source_calibration_status: str
        - claim_level: str
    """
    specs = standard_receptor_specs()
    result = {}
    for name, spec in specs.items():
        result[name] = {
            "receptor_index": spec.receptor_index,
            "sign": spec.sign,
            "tau_ms": spec.tau_ms,
            "reversal_mV": spec.reversal_mV,
            "source_calibration_status": spec.source_calibration_status,
            "claim_level": spec.claim_level,
        }
    return result


def diagnose_connectivity(
    eig_network: EIGNetwork,
    edges: EdgeList,
) -> dict[str, Any]:
    """Generate diagnostic statistics about network connectivity.

    Parameters
    ----------
    eig_network : EIGNetwork
        Network to analyze.

    edges : EdgeList
        Connectivity to analyze.

    Returns
    -------
    dict
        Statistics including:
        - n_neurons: int
        - n_edges: int
        - connection_density: float
        - receptor_counts: dict
        - edge_weight_stats: dict (mean, std, min, max)
        - excitatory_fraction: float
        - cell_type_distribution: dict
    """
    n_neurons = len(eig_network.params.a)
    n_edges = len(edges.pre)
    max_edges = n_neurons * (n_neurons - 1)
    connection_density = n_edges / max_edges if max_edges > 0 else 0.0

    # Receptor counts
    receptor_counts = {}
    for i in range(4):
        count = (edges.receptor_index == i).sum()
        if count > 0:
            receptor_names = ["AMPA", "GABA_A", "NMDA", "GABA_B"]
            receptor_counts[receptor_names[i]] = int(count)

    # Weight statistics
    weights = np.asarray(edges.weight)
    edge_weight_stats = {
        "mean": float(np.mean(np.abs(weights))),
        "std": float(np.std(weights)),
        "min": float(np.min(weights)),
        "max": float(np.max(weights)),
        "n_nonzero": int(np.count_nonzero(weights)),
    }

    # Cell type distribution
    labels_arr = np.array(eig_network.params.labels)
    cell_types = {}
    for ct in np.unique(labels_arr):
        cell_types[ct] = int(np.sum(labels_arr == ct))

    # Excitatory fraction
    exc_fraction = cell_types.get("E", 0) / n_neurons if n_neurons > 0 else 0.0

    return {
        "n_neurons": n_neurons,
        "n_edges": n_edges,
        "connection_density": float(connection_density),
        "receptor_counts": receptor_counts,
        "edge_weight_stats": edge_weight_stats,
        "excitatory_fraction": float(exc_fraction),
        "cell_type_distribution": cell_types,
    }


def project_to_laminar_field(
    source_proxy: Array,  # (n_steps, n_neurons) or (n_neurons, n_steps)
    positions: Array,  # (n_neurons, 3) with normalized depth in [0, 1]
    *,
    n_contacts: int = 16,
    width: float = 0.1,
    dtype: str = "float32",
) -> jaxfne.FieldOutput:
    """Project source traces to laminar field using jaxfne's Gaussian proxy.

    Parameters
    ----------
    source_proxy : Array
        Source traces, shape (n_steps, n_neurons) or (n_neurons, n_steps).
        If (n_neurons, n_steps), will be transposed to (n_steps, n_neurons).

    positions : Array
        Neuron positions (n_neurons, 3). Third coordinate should be laminar
        depth normalized to [0, 1].

    n_contacts : int, default 16
        Number of laminar recording contacts.

    width : float, default 0.1
        Gaussian width in relative laminar-depth units.

    dtype : str
        JAX dtype.

    Returns
    -------
    FieldOutput
        Includes: source_proxy, phi_e_proxy, csd_proxy, lfp_proxy, kernel,
                  contact_depths, diagnostics.
    """
    # Ensure (n_steps, n_neurons) shape
    # jaxfne expects (n_steps, n_neurons); positions is (n_neurons, 3)
    n_neurons = positions.shape[0]
    if source_proxy.shape[1] == n_neurons:
        # Already (n_steps, n_neurons)
        pass
    elif source_proxy.shape[0] == n_neurons:
        # (n_neurons, n_steps) - transpose
        source_proxy = source_proxy.T
    else:
        raise ValueError(
            f"source_proxy shape {source_proxy.shape} incompatible with "
            f"positions shape {positions.shape}"
        )

    return jaxfne.project_laminar_sources(
        source_proxy.astype(dtype),
        positions.astype(dtype),
        n_contacts=n_contacts,
        width=width,
        dtype=dtype,
    )
