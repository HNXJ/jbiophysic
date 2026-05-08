"""Izhikevich point-neuron primitives.

Native variables are mV/ms-scaled and phenomenological. The input `I` is not automatically
an SI membrane current; use an explicit calibration layer before coupling to TFNE sources.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

Array = jax.Array


class IzhikevichParams(NamedTuple):
    a: float = 0.02
    b: float = 0.2
    c: float = -65.0
    d: float = 8.0
    v_spike_mV: float = 30.0


REGULAR_SPIKING = IzhikevichParams(a=0.02, b=0.2, c=-65.0, d=8.0)
FAST_SPIKING = IzhikevichParams(a=0.1, b=0.2, c=-65.0, d=2.0)
LOW_THRESHOLD_SPIKING = IzhikevichParams(a=0.02, b=0.25, c=-65.0, d=2.0)


def izhikevich_step(v_mV: Array, u: Array, I: Array, params: IzhikevichParams, dt_ms: float) -> tuple[Array, Array, Array]:
    """Advance one Euler step and apply reset.

    Returns `(v_next_mV, u_next, spiked_bool)`.
    """
    if dt_ms <= 0:
        raise ValueError("dt_ms must be positive")
    dv = 0.04 * v_mV**2 + 5.0 * v_mV + 140.0 - u + I
    du = params.a * (params.b * v_mV - u)
    v_next = v_mV + dt_ms * dv
    u_next = u + dt_ms * du
    spiked = v_next >= params.v_spike_mV
    v_next = jnp.where(spiked, params.c, v_next)
    u_next = jnp.where(spiked, u_next + params.d, u_next)
    return v_next, u_next, spiked


def simulate_izhikevich(
    current: Array,
    *,
    params: IzhikevichParams = REGULAR_SPIKING,
    dt_ms: float = 0.5,
    v0_mV: float = -65.0,
    u0: float | None = None,
) -> tuple[Array, Array, Array]:
    """Simulate a scalar Izhikevich neuron for a current trace.

    Parameters
    ----------
    current:
        One-dimensional current-like drive in native Izhikevich units.
    """
    current = jnp.asarray(current)
    if current.ndim != 1:
        raise ValueError("current must be one-dimensional")
    if u0 is None:
        u0 = params.b * v0_mV

    def body(carry, I_t):
        v, u = carry
        v_next, u_next, spiked = izhikevich_step(v, u, I_t, params, dt_ms)
        return (v_next, u_next), (v_next, u_next, spiked)

    (_vf, _uf), ys = jax.lax.scan(body, (jnp.asarray(v0_mV), jnp.asarray(u0)), current)
    v, u, spikes = ys
    return v, u, spikes
