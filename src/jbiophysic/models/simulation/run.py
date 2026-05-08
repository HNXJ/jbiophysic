# src/jbiophysic/models/simulation/run.py
from __future__ import annotations

from typing import Any

import jax.numpy as jnp

from jbiophysic.common.types.simulation import SimulationConfig, SimulationResult
from jbiophysic.common.utils.logging import get_logger

logger = get_logger(__name__)

try:
    import jaxley as jx
except ModuleNotFoundError:  # pragma: no cover - exercised when jaxley is absent
    jx = None


def _fallback_trace(brain: Any, config: SimulationConfig) -> jnp.ndarray:
    n_nodes = len(getattr(brain, "nodes", list(getattr(brain, "cells", []))))
    n_steps = int(round(config.t_max / config.dt)) + 1
    t = jnp.arange(n_steps) * config.dt
    # Deterministic finite smoke trace: near-rest membrane potential with tiny node-specific ripple.
    offsets = jnp.arange(max(n_nodes, 1)).reshape(-1, 1) * 0.01
    return -65.0 + offsets + 0.01 * jnp.sin(2.0 * jnp.pi * t.reshape(1, -1) / max(config.t_max, config.dt))


def run_simulation(
    brain: Any,
    config: SimulationConfig,
    params: dict[str, Any] | None = None,
) -> SimulationResult:
    """Run a short cortical-model simulation.

    Jaxley networks use Jaxley integration.  Lightweight ``SimpleNetwork`` fallbacks return a
    deterministic finite trace, preserving import and smoke-test behavior without pretending to
    be a validated biological simulation.
    """
    logger.info(f"Running simulation for T={config.t_max}ms with dt={config.dt}ms")

    if config.dt <= 0 or config.t_max < 0:
        raise ValueError("config.dt must be positive and config.t_max must be nonnegative")

    if jx is not None and brain.__class__.__module__.startswith("jaxley"):
        if brain.recordings.empty:
            brain.record("v")
        v_trace = jx.integrate(brain, t_max=config.t_max, delta_t=config.dt)
        backend = "jaxley"
    else:
        if hasattr(brain, "record") and getattr(getattr(brain, "recordings", None), "empty", False):
            brain.record("v")
        v_trace = _fallback_trace(brain, config)
        backend = "simple_fallback"

    logger.info("Integration successful.")
    return SimulationResult(
        v_trace=v_trace,
        currents=None,
        state=None,
        metadata={"t_max": config.t_max, "dt": config.dt, "seed": config.seed, "backend": backend},
    )
