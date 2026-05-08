"""Sparse edge-list JAX backend for efficient Izhikevich simulations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

import jax
import jax.numpy as jnp

Array = jax.Array


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class EdgeList:
    """Sparse connectivity representation for JAX simulation."""

    pre: Array
    post: Array
    weight: Array
    receptor_index: Array
    delay_steps: Array
    plasticity_scale: Array

    def tree_flatten(self):
        children = (
            self.pre,
            self.post,
            self.weight,
            self.receptor_index,
            self.delay_steps,
            self.plasticity_scale,
        )
        aux_data = None
        return (children, aux_data)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children)


class IzhikevichState(NamedTuple):
    v: Array
    u: Array
    syn_state: Array
    weights: Array


def simulate_izhikevich_edge_jax(
    params: tuple[Array, Array, Array, Array],
    state0: IzhikevichState,
    edges: EdgeList,
    drives: Array,
    dt_ms: float,
    noise_sd: float = 0.5,
    key: Array | None = None,
    plasticity_enabled: bool = True,
    plasticity_lr: float = 1e-4,
    weight_max: float = 0.25,
) -> tuple[IzhikevichState, tuple[Array, Array]]:
    """Simulate a network using sparse JAX ops.

    Parameters
    ----------
    params: (a, b, c, d) vectorized parameters.
    drives: [steps, n_neurons] drive matrix.
    """
    a, b, c, d = params
    steps = drives.shape[0]
    n = drives.shape[1]
    if key is None:
        key = jax.random.PRNGKey(0)

    # Simplified fixed decay per receptor for now
    # AMPA: 2ms, GABA: 10ms, NMDA: 100ms
    tau = jnp.array([2.0, 10.0, 100.0])
    decay = jnp.exp(-dt_ms / tau[edges.receptor_index])
    sign = jnp.where(edges.receptor_index == 1, -1.0, 1.0) # Assume 1 is GABA

    def body(carry, drive_t_key):
        state, (drive_t, step_key) = carry, drive_t_key
        v, u, syn_state, weights = state

        # Compute synaptic current: I_syn = sum_pre (sign * weight * syn_state)
        # We use jax.ops.segment_sum for efficient sparse aggregation
        I_syn = jax.ops.segment_sum(sign * weights * syn_state, edges.post, num_segments=n)

        noise = noise_sd * jax.random.normal(step_key, (n,))
        current_in = drive_t + I_syn + noise

        dv = 0.04 * v**2 + 5.0 * v + 140.0 - u + current_in
        du = a * (b * v - u)
        v_next_pre = v + dt_ms * dv
        u_next_pre = u + dt_ms * du
        
        spiked = v_next_pre >= 30.0
        v_next = jnp.where(spiked, c, v_next_pre)
        u_next = jnp.where(spiked, u_next_pre + d, u_next_pre)

        # Update synaptic state: spiked[pre] increments it
        syn_next = syn_state * decay + spiked[edges.pre].astype(jnp.float32)

        # Plasticity: simple Hebbian-like
        if plasticity_enabled:
            hebb = spiked[edges.pre].astype(jnp.float32) * spiked[edges.post].astype(jnp.float32)
            delta_w = plasticity_lr * edges.plasticity_scale * hebb * (weight_max - weights)
            weights_next = jnp.clip(weights + delta_w, 0.0, weight_max)
        else:
            weights_next = weights

        next_state = IzhikevichState(v_next, u_next, syn_next, weights_next)
        return next_state, (v_next, spiked)

    keys = jax.random.split(key, steps)
    final_state, (V, spikes) = jax.lax.scan(
        body, 
        state0,
        (drives, keys)
    )
    return final_state, (V, spikes)
