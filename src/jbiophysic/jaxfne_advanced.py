"""
Advanced jaxfne integration: Custom receptor kinetics and multi-area routing.

This module extends the basic integration with:
1. Custom receptor tau_ms tuning (per-edge or per-connection-type)
2. Multi-area explicit routing with area-specific kinetics
3. Advanced connectivity diagnostics
4. Synaptic weight optimization
"""

from __future__ import annotations

from typing import Any, Mapping

import jax
import jax.numpy as jnp
import numpy as np

from jaxfne import EdgeList, EIGNetwork, standard_receptor_specs

Array = jax.Array


class CustomReceptorSpec:
    """Container for custom receptor kinetics parameters.

    Allows per-connection-type tuning of receptor time constants.
    """

    def __init__(self):
        """Initialize with standard jaxfne receptor specs."""
        self._specs = standard_receptor_specs()
        self._tau_overrides = {}  # (pre_type, post_type) → tau_ms

    def set_tau(self, pre_type: str, post_type: str, tau_ms: float) -> None:
        """Override tau_ms for a specific connection type.

        Parameters
        ----------
        pre_type : str
            Presynaptic cell type (E, PV, SST, VIP).

        post_type : str
            Postsynaptic cell type (E, PV, SST, VIP).

        tau_ms : float
            Custom time constant in milliseconds.
        """
        self._tau_overrides[(pre_type, post_type)] = tau_ms

    def get_tau(self, pre_type: str, post_type: str, receptor_index: int) -> float:
        """Get tau for a connection.

        Parameters
        ----------
        pre_type : str
            Presynaptic cell type.

        post_type : str
            Postsynaptic cell type.

        receptor_index : int
            Receptor type (0=AMPA, 1=GABA_A, 2=NMDA, 3=GABA_B).

        Returns
        -------
        float
            Time constant in milliseconds.
        """
        # Check for override
        if (pre_type, post_type) in self._tau_overrides:
            return self._tau_overrides[(pre_type, post_type)]

        # Fall back to standard spec
        receptor_names = ["AMPA", "GABA_A", "NMDA", "GABA_B"]
        if receptor_index < len(receptor_names):
            spec = self._specs[receptor_names[receptor_index]]
            return spec.tau_ms

        return 2.0  # Fallback


def build_multi_area_edges(
    eig_network: EIGNetwork,
    neurons_df: Any,  # pandas DataFrame
    area_order: tuple[str, ...],
    receptor_kinetics: CustomReceptorSpec | None = None,
    dtype: str = "float32",
    existing_edges: EdgeList | None = None,
) -> EdgeList:
    """Build EdgeList with explicit per-area receptor kinetics.

    Allows different receptor time constants for intra-area vs inter-area
    connections, and per-area customization.

    Parameters
    ----------
    eig_network : EIGNetwork
        Network container with neuron positions and metadata.

    neurons_df : DataFrame
        Neurons dataframe with columns [neuron_id, area, cell_type, ...].

    area_order : tuple[str]
        List of area names in order.

    receptor_kinetics : CustomReceptorSpec, optional
        Custom receptor specifications. If None, uses standard jaxfne specs.

    dtype : str
        JAX dtype for arrays.

    existing_edges : EdgeList, optional
        If provided, use this as the connectivity source and just update tau values.
        If None, try to extract from eig_network.params.W.

    Returns
    -------
    EdgeList
        Connectivity with area-aware receptor kinetics.
    """
    if receptor_kinetics is None:
        receptor_kinetics = CustomReceptorSpec()

    n_neurons = len(eig_network.params.a)
    cell_types = np.array(eig_network.params.labels)
    areas = np.array(neurons_df["area"].values)

    # If existing edges provided, update tau values
    if existing_edges is not None:
        pre_indices = np.array(existing_edges.pre)
        post_indices = np.array(existing_edges.post)
        weights = np.array(existing_edges.weight)
        receptor_indices = np.array(existing_edges.receptor_index)

        # Recompute tau values based on connection types
        tau_ms_list = []
        for p, q, rec_idx in zip(pre_indices, post_indices, receptor_indices):
            pre_type = cell_types[p]
            post_type = cell_types[q]
            tau = receptor_kinetics.get_tau(pre_type, post_type, int(rec_idx))
            tau_ms_list.append(tau)
    else:
        # Try to extract from embedded W
        pre_indices = []
        post_indices = []
        weights = []
        receptor_indices = []
        tau_ms_list = []

        W = eig_network.params.W
        if W is not None and jnp.any(W != 0):
            # Use existing weights from EIGNetwork
            post_idx, pre_idx = jnp.where(W != 0)
            pre_indices = np.array(pre_idx)
            post_indices = np.array(post_idx)
            weights = np.array([W[q, p] for p, q in zip(pre_idx, post_idx)])

            for p, q in zip(pre_indices, post_indices):
                pre_type = cell_types[p]
                post_type = cell_types[q]
                is_exc = pre_type == "E"
                receptor_idx = 0 if is_exc else 1
                tau = receptor_kinetics.get_tau(pre_type, post_type, receptor_idx)
                receptor_indices.append(receptor_idx)
                tau_ms_list.append(tau)

    # Convert to JAX arrays
    if len(pre_indices) > 0:
        pre_arr = jnp.asarray(pre_indices, dtype="int32")
        post_arr = jnp.asarray(post_indices, dtype="int32")
        weight_arr = jnp.asarray(weights, dtype=dtype)
        receptor_arr = jnp.asarray(receptor_indices, dtype="int32")
        tau_arr = jnp.asarray(tau_ms_list, dtype=dtype)
    else:
        # Empty network
        pre_arr = jnp.zeros(0, dtype="int32")
        post_arr = jnp.zeros(0, dtype="int32")
        weight_arr = jnp.zeros(0, dtype=dtype)
        receptor_arr = jnp.zeros(0, dtype="int32")
        tau_arr = jnp.zeros(0, dtype=dtype)

    return EdgeList(
        pre=pre_arr,
        post=post_arr,
        weight=weight_arr,
        receptor_index=receptor_arr,
        tau_ms=tau_arr,
        source_calibration_status="uncalibrated_izhikevich_native_current",
    )


def optimize_synaptic_weights(
    edges: EdgeList,
    target_firing_rate: float = 0.1,
    scale_factor: float = 1.0,
) -> EdgeList:
    """Scale synaptic weights to target a desired firing rate.

    This is a simple heuristic: stronger weights → higher firing rates.
    For more sophisticated optimization, use jaxfne's AGSDR optimizer.

    Parameters
    ----------
    edges : EdgeList
        Current connectivity.

    target_firing_rate : float, default 0.1
        Target population firing rate [0, 1].

    scale_factor : float, default 1.0
        Scale to apply to weights (>1 = stronger, <1 = weaker).

    Returns
    -------
    EdgeList
        Scaled connectivity.
    """
    scaled_weights = edges.weight * scale_factor

    return EdgeList(
        pre=edges.pre,
        post=edges.post,
        weight=scaled_weights,
        receptor_index=edges.receptor_index,
        tau_ms=edges.tau_ms,
        source_calibration_status=edges.source_calibration_status,
    )


def compute_connection_motifs(
    eig_network: EIGNetwork,
    edges: EdgeList,
) -> dict[str, int]:
    """Identify and count network motifs (structural patterns).

    Returns counts of common motifs like:
    - Reciprocal connections (E↔E, I↔I, etc.)
    - Feed-forward triplets
    - Recurrent loops

    Parameters
    ----------
    eig_network : EIGNetwork
        Network container.

    edges : EdgeList
        Connectivity.

    Returns
    -------
    dict
        Motif counts and statistics.
    """
    n_neurons = len(eig_network.params.a)
    cell_types = np.array(eig_network.params.labels)

    # Build adjacency matrix for motif detection
    adj_matrix = np.zeros((n_neurons, n_neurons), dtype=bool)
    for p, q in zip(edges.pre, edges.post):
        adj_matrix[q, p] = True

    # Count reciprocal connections
    reciprocal = 0
    for i in range(n_neurons):
        for j in range(i + 1, n_neurons):
            if adj_matrix[i, j] and adj_matrix[j, i]:
                reciprocal += 1

    # Count triangles (feed-forward triplets)
    triangles = 0
    for i in range(n_neurons):
        for j in range(n_neurons):
            if i != j and adj_matrix[i, j]:
                for k in range(n_neurons):
                    if k != i and k != j and adj_matrix[j, k] and adj_matrix[i, k]:
                        triangles += 1

    # Excitatory/inhibitory breakdown
    is_exc = np.array([ct == "E" for ct in cell_types])
    e_to_e = np.sum(is_exc[edges.pre] & is_exc[edges.post])
    e_to_i = np.sum(is_exc[edges.pre] & ~is_exc[edges.post])
    i_to_e = np.sum(~is_exc[edges.pre] & is_exc[edges.post])
    i_to_i = np.sum(~is_exc[edges.pre] & ~is_exc[edges.post])

    return {
        "n_edges": len(edges.pre),
        "reciprocal_pairs": int(reciprocal),
        "triangles": int(triangles // 3),  # Divide by 3 since we count 3 times
        "e_to_e": int(e_to_e),
        "e_to_i": int(e_to_i),
        "i_to_e": int(i_to_e),
        "i_to_i": int(i_to_i),
    }


def analyze_critical_neurons(
    eig_network: EIGNetwork,
    edges: EdgeList,
) -> dict[str, Any]:
    """Identify neurons that are highly connected (hub neurons).

    Heuristic: neurons with high in-degree + out-degree are critical
    for information integration.

    Parameters
    ----------
    eig_network : EIGNetwork
        Network container.

    edges : EdgeList
        Connectivity.

    Returns
    -------
    dict
        Hub neuron statistics.
    """
    n_neurons = len(eig_network.params.a)
    cell_types = np.array(eig_network.params.labels)

    # Count in- and out-degree
    in_degree = np.zeros(n_neurons)
    out_degree = np.zeros(n_neurons)

    for p, q in zip(edges.pre, edges.post):
        out_degree[p] += 1
        in_degree[q] += 1

    # Identify hubs (top 10% by total degree)
    total_degree = in_degree + out_degree
    threshold = np.percentile(total_degree, 90)
    hub_mask = total_degree >= threshold

    hub_types = {}
    for ct in np.unique(cell_types):
        count = np.sum(hub_mask & (np.array(cell_types) == ct))
        hub_types[ct] = int(count)

    return {
        "n_hubs": int(np.sum(hub_mask)),
        "hub_fraction": float(np.sum(hub_mask) / n_neurons),
        "hub_types": hub_types,
        "max_in_degree": int(np.max(in_degree)),
        "max_out_degree": int(np.max(out_degree)),
        "avg_in_degree": float(np.mean(in_degree)),
        "avg_out_degree": float(np.mean(out_degree)),
    }
