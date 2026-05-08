"""Conductance-based synapse helpers."""

from __future__ import annotations

import jax.numpy as jnp


def conductance_current(g: jnp.ndarray, s: jnp.ndarray, v_mV: jnp.ndarray, e_rev_mV: float) -> jnp.ndarray:
    """Return synaptic current in conductance units using `I = g*s*(V-E)`."""
    return g * s * (v_mV - e_rev_mV)


def exp_synapse_step(s: jnp.ndarray, spike_drive: jnp.ndarray, tau_ms: float, dt_ms: float, amplitude: float = 1.0) -> jnp.ndarray:
    """First-order exponential synapse state update."""
    if tau_ms <= 0 or dt_ms <= 0:
        raise ValueError("tau_ms and dt_ms must be positive")
    return jnp.clip(s + dt_ms * (-s / tau_ms + amplitude * spike_drive * (1.0 - s)), 0.0, 1.0)
