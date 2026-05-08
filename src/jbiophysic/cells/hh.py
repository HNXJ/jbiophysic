"""Minimal Hodgkin-Huxley membrane primitive in classic mV/ms units."""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

Array = jax.Array


class HHParams(NamedTuple):
    C_m_uF_cm2: float = 1.0
    g_Na_mS_cm2: float = 120.0
    g_K_mS_cm2: float = 36.0
    g_L_mS_cm2: float = 0.3
    E_Na_mV: float = 50.0
    E_K_mV: float = -77.0
    E_L_mV: float = -54.387


def _safe_expm1_ratio(x: Array, scale: float) -> Array:
    """Return x / (1 - exp(-x / scale)) with a stable small-x limit."""
    z = x / scale
    return jnp.where(jnp.abs(z) < 1e-6, scale * (1.0 + z / 2.0), x / (1.0 - jnp.exp(-z)))


def alpha_m(V_mV: Array) -> Array:
    return 0.1 * _safe_expm1_ratio(V_mV + 40.0, 10.0)


def beta_m(V_mV: Array) -> Array:
    return 4.0 * jnp.exp(-(V_mV + 65.0) / 18.0)


def alpha_h(V_mV: Array) -> Array:
    return 0.07 * jnp.exp(-(V_mV + 65.0) / 20.0)


def beta_h(V_mV: Array) -> Array:
    return 1.0 / (1.0 + jnp.exp(-(V_mV + 35.0) / 10.0))


def alpha_n(V_mV: Array) -> Array:
    return 0.01 * _safe_expm1_ratio(V_mV + 55.0, 10.0)


def beta_n(V_mV: Array) -> Array:
    return 0.125 * jnp.exp(-(V_mV + 65.0) / 80.0)


def steady_state_gates(V_mV: Array) -> tuple[Array, Array, Array]:
    """Return HH gate steady states `(m_inf, h_inf, n_inf)`."""
    am, bm = alpha_m(V_mV), beta_m(V_mV)
    ah, bh = alpha_h(V_mV), beta_h(V_mV)
    an, bn = alpha_n(V_mV), beta_n(V_mV)
    return am / (am + bm), ah / (ah + bh), an / (an + bn)


def hh_currents(V_mV: Array, m: Array, h: Array, n: Array, params: HHParams) -> tuple[Array, Array, Array]:
    """Return `(I_Na, I_K, I_L)` in uA/cm^2 using outward-positive convention."""
    I_Na = params.g_Na_mS_cm2 * (m**3) * h * (V_mV - params.E_Na_mV)
    I_K = params.g_K_mS_cm2 * (n**4) * (V_mV - params.E_K_mV)
    I_L = params.g_L_mS_cm2 * (V_mV - params.E_L_mV)
    return I_Na, I_K, I_L


def hh_step(
    V_mV: Array,
    m: Array,
    h: Array,
    n: Array,
    I_app_uA_cm2: Array,
    params: HHParams,
    dt_ms: float,
) -> tuple[Array, Array, Array, Array]:
    """Euler HH step for smoke-scale tests."""
    if dt_ms <= 0:
        raise ValueError("dt_ms must be positive")
    I_Na, I_K, I_L = hh_currents(V_mV, m, h, n, params)
    dV = (I_app_uA_cm2 - I_Na - I_K - I_L) / params.C_m_uF_cm2
    dm = alpha_m(V_mV) * (1.0 - m) - beta_m(V_mV) * m
    dh = alpha_h(V_mV) * (1.0 - h) - beta_h(V_mV) * h
    dn = alpha_n(V_mV) * (1.0 - n) - beta_n(V_mV) * n
    return V_mV + dt_ms * dV, m + dt_ms * dm, h + dt_ms * dh, n + dt_ms * dn


def simulate_hh(
    current_uA_cm2: Array,
    *,
    params: HHParams | None = None,
    dt_ms: float = 0.01,
    V0_mV: float = -65.0,
) -> tuple[Array, Array]:
    """Simulate a single HH compartment.

    Returns `(V_mV, gates)` where gates has shape `[3, T]` for m, h, n.
    This eager loop is intentional for fast smoke tests; production JAX pipelines can wrap
    `hh_step` in `lax.scan` once solver tolerances and stiffness policy are fixed.
    """
    if params is None:
        params = HHParams()
    current_uA_cm2 = jnp.asarray(current_uA_cm2)
    if current_uA_cm2.ndim != 1:
        raise ValueError("current_uA_cm2 must be one-dimensional")
    V = jnp.asarray(V0_mV)
    m, h, n = steady_state_gates(V)
    v_hist = []
    m_hist = []
    h_hist = []
    n_hist = []
    for I_t in current_uA_cm2:
        V, m, h, n = hh_step(V, m, h, n, I_t, params, dt_ms)
        v_hist.append(V)
        m_hist.append(m)
        h_hist.append(h)
        n_hist.append(n)
    return jnp.stack(v_hist), jnp.stack([jnp.stack(m_hist), jnp.stack(h_hist), jnp.stack(n_hist)], axis=0)
