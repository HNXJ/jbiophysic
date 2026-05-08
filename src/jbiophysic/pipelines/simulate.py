"""Simulation pipeline smoke wrappers."""

from __future__ import annotations

import jax.numpy as jnp

from jbiophysic.cells.izhikevich import IzhikevichParams, simulate_izhikevich


def run_izhikevich_constant_current(T_ms: float = 100.0, dt_ms: float = 0.5, I: float = 10.0) -> dict[str, object]:
    """Run a deterministic Izhikevich smoke simulation."""
    steps = int(T_ms / dt_ms)
    current = jnp.full((steps,), I)
    v, u, spikes = simulate_izhikevich(current, params=IzhikevichParams(), dt_ms=dt_ms)
    return {"v_mV": v, "u": u, "spikes": spikes, "n_spikes": int(jnp.sum(spikes))}
