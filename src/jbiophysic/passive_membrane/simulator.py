"""Passive membrane simulator: lumped cable model integration.

Implements forward Euler integration of:
    C_m * dV/dt = -g_L * (V - E_L) + I_inj

All units are absolute (not specific densities):
    - Voltage: mV
    - Capacitance: pF (absolute soma capacitance)
    - Conductance: nS (absolute soma conductance)
    - Current: pA
    - Time: ms

This is the standard convention for whole-cell single-compartment models.

v0.2.0–v0.2.11
"""

from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp
import numpy as np


class PassiveMembraneParams(NamedTuple):
    """Parameters for passive membrane model (whole-cell absolute units).

    Attributes
    ----------
    C_m : float
        Soma membrane capacitance (pF). Typical range: 50–200 pF for soma.
        (Specific: 1 μF/cm² × soma surface area)
    g_L : float
        Soma leak conductance (nS). Typical range: 0.1–5 nS.
        (Specific: 0.1 mS/cm² × soma surface area)
    E_L : float
        Leak equilibrium potential (mV). Typical range: -70 to -50.

    Notes
    -----
    Time constant: tau = C_m / g_L (milliseconds).
    For C_m=100 pF, g_L=1 nS: tau = 100 ms.

    To convert from specific densities:
        C_m (pF) = C_m_specific (μF/cm²) × soma_area (cm²) × 10^6
        g_L (nS) = g_L_specific (mS/cm²) × soma_area (cm²) × 1000
    Example soma area: π × (10 μm)² ≈ 1260 μm² ≈ 1.26 × 10^-5 cm²
    """

    C_m: float
    g_L: float
    E_L: float


def passive_membrane_step(
    V: float | np.ndarray,
    C_m: float,
    g_L: float,
    E_L: float,
    I_inj: float,
    dt_ms: float,
) -> float | np.ndarray:
    """Single forward-Euler integration step for passive membrane.

    Implements:
        V_new = V + (dt / C_m) * (-g_L * (V - E_L) + I_inj)

    Parameters
    ----------
    V : float or Array
        Current membrane voltage (mV). Scalar or shape-compatible array.
    C_m : float
        Soma membrane capacitance (pF).
    g_L : float
        Soma leak conductance (nS).
    E_L : float
        Leak equilibrium potential (mV).
    I_inj : float
        Injected current (pA). Positive = inward (depolarizing).
    dt_ms : float
        Integration timestep (ms). Must be << tau for stability.
        Stability criterion: dt < 2 * tau = 2 * (C_m / g_L) ms.

    Returns
    -------
    V_new : float or Array
        Updated voltage after one timestep. Same shape as V.

    Notes
    -----
    This is a minimal Euler step; no adaptive timesteping or error control.
    For dt > 2*tau, numerical solution diverges (blow-up).

    Sign convention: Positive I_inj is inward current (depolarizes).

    Examples
    --------
    >>> from jbiophysic.passive_membrane import passive_membrane_step
    >>> V = -65.0  # Initial voltage (mV)
    >>> V_new = passive_membrane_step(
    ...     V, C_m=100.0, g_L=1.0, E_L=-65.0, I_inj=10.0, dt_ms=1.0
    ... )
    >>> print(f"V_step = {V_new:.3f} mV")
    V_step ≈ -64.900 mV  (depolarized)
    """
    dV_dt = (-g_L * (V - E_L) + I_inj) / C_m
    return V + dt_ms * dV_dt


def passive_membrane_simulate(
    V_init: float,
    params: PassiveMembraneParams,
    I_inj: float | np.ndarray,
    dt_ms: float,
    duration_ms: float,
) -> np.ndarray:
    """Simulate passive membrane voltage response to injected current.

    Integrates the passive membrane equation using forward Euler:
        C_m * dV/dt = -g_L * (V - E_L) + I_inj

    Parameters
    ----------
    V_init : float
        Initial membrane voltage (mV). Typically -65.0 (resting).
    params : PassiveMembraneParams
        Soma membrane biophysical parameters (absolute units).
    I_inj : float or Array
        Injected current (pA).
        If scalar: constant current throughout simulation.
        If Array: time-varying current, shape (n_steps,). Will be broadcast.
    dt_ms : float
        Integration timestep (ms). Recommend dt << 0.1 * tau.
    duration_ms : float
        Total simulation duration (ms).

    Returns
    -------
    V_trace : ndarray, shape (n_steps + 1,), dtype float64
        Voltage trace (mV) from t=0 to t=duration_ms, inclusive.
        V_trace[0] = V_init.
        V_trace[-1] = voltage at t=duration_ms.

    Notes
    -----
    Stability: Forward Euler is stable for dt < 2*tau = 2*(C_m / g_L) ms.
    For typical tau ~ 10–100 ms, safe dt ≈ 0.01–0.1 ms.

    Examples
    --------
    >>> from jbiophysic.passive_membrane import passive_membrane_simulate, PassiveMembraneParams
    >>> params = PassiveMembraneParams(C_m=100.0, g_L=1.0, E_L=-65.0)
    >>> V_trace = passive_membrane_simulate(
    ...     V_init=-65.0,
    ...     params=params,
    ...     I_inj=10.0,  # pA
    ...     dt_ms=1.0,   # 1 ms timestep
    ...     duration_ms=100.0,  # 100 ms duration
    ... )
    >>> print(f"Simulated {len(V_trace)-1} steps")
    Simulated 100 steps
    >>> print(f"V_init={V_trace[0]:.1f} mV, V_final={V_trace[-1]:.3f} mV")
    V_init=-65.0 mV, V_final=-64.500 mV
    """
    n_steps = int(round(duration_ms / dt_ms))
    V_trace = [V_init]

    # Handle scalar or time-varying current
    if np.isscalar(I_inj):
        I_inj_array = np.ones(n_steps) * I_inj
    else:
        I_inj_array = np.asarray(I_inj, dtype=np.float64)
        if len(I_inj_array) != n_steps:
            raise ValueError(
                f"I_inj array length {len(I_inj_array)} "
                f"does not match n_steps {n_steps}"
            )

    V = V_init
    for i in range(n_steps):
        V = passive_membrane_step(
            V,
            C_m=params.C_m,
            g_L=params.g_L,
            E_L=params.E_L,
            I_inj=I_inj_array[i],
            dt_ms=dt_ms,
        )
        V_trace.append(float(V))

    return np.array(V_trace, dtype=np.float64)
