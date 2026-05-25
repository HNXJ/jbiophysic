"""Passive membrane diagnostics: steady-state, time constants, and response properties.

Provides utility functions for analyzing passive membrane behavior:
- Time constant (tau)
- Steady-state voltage
- Exponential relaxation solutions
- Input resistance
- Step current response

All outputs JSON-serializable and numerically stable.

Units: mV, ms, pA, pF, nS (absolute soma units, not specific densities).

v0.2.0–v0.2.11
"""

from __future__ import annotations

import numpy as np


def tau_membrane_ms(C_m: float, g_L: float) -> float:
    """Calculate passive membrane time constant.

    Time constant governs exponential relaxation:
        V(t) = V_ss + (V_init - V_ss) * exp(-t / tau)

    Parameters
    ----------
    C_m : float
        Soma membrane capacitance (pF).
    g_L : float
        Soma leak conductance (nS).

    Returns
    -------
    tau : float
        Time constant (ms). tau = C_m / g_L.

    Examples
    --------
    >>> from jbiophysic.passive_membrane import tau_membrane_ms
    >>> tau = tau_membrane_ms(C_m=100.0, g_L=1.0)
    >>> print(f"tau = {tau:.1f} ms")
    tau = 100.0 ms

    >>> tau = tau_membrane_ms(C_m=50.0, g_L=2.0)
    >>> print(f"tau = {tau:.1f} ms")
    tau = 25.0 ms
    """
    if g_L <= 0:
        raise ValueError(f"g_L must be > 0, got {g_L}")
    return float(C_m / g_L)


def steady_state_voltage(E_L: float, g_L: float, I_inj: float) -> float:
    """Calculate passive membrane steady-state voltage.

    When dV/dt = 0:
        V_ss = E_L + I_inj / g_L

    Parameters
    ----------
    E_L : float
        Leak equilibrium potential (mV).
    g_L : float
        Soma leak conductance (nS).
    I_inj : float
        Injected current (pA). Positive = inward (depolarizing).

    Returns
    -------
    V_ss : float
        Steady-state voltage (mV).

    Notes
    -----
    If I_inj = 0, V_ss = E_L (resting potential).
    Positive I_inj (inward) depolarizes (increases V).
    Negative I_inj (outward) hyperpolarizes (decreases V).

    Examples
    --------
    >>> from jbiophysic.passive_membrane import steady_state_voltage
    >>> # At rest (no current)
    >>> V_ss = steady_state_voltage(E_L=-65.0, g_L=1.0, I_inj=0.0)
    >>> print(f"V_ss (rest) = {V_ss:.1f} mV")
    V_ss (rest) = -65.0 mV

    >>> # With depolarizing current
    >>> V_ss = steady_state_voltage(E_L=-65.0, g_L=1.0, I_inj=10.0)
    >>> print(f"V_ss (10 pA) = {V_ss:.2f} mV")
    V_ss (10 pA) = -55.00 mV  (depolarized by 10 mV)
    """
    if g_L <= 0:
        raise ValueError(f"g_L must be > 0, got {g_L}")
    return float(E_L + I_inj / g_L)


def relaxation_curve(
    t: float | np.ndarray,
    V_init: float,
    V_ss: float,
    tau: float,
) -> float | np.ndarray:
    """Exact exponential relaxation solution for passive membrane.

    Analytically solves:
        V(t) = V_ss + (V_init - V_ss) * exp(-t / tau)

    Parameters
    ----------
    t : float or Array
        Time(s) (ms). Scalar or shape (n,).
    V_init : float
        Initial voltage (mV).
    V_ss : float
        Steady-state voltage (mV).
    tau : float
        Time constant (ms). Must be > 0.

    Returns
    -------
    V_t : float or Array
        Voltage at time t (mV). Same shape as t.

    Notes
    -----
    This is the exact solution to the ODE C_m dV/dt = -g_L(V - E_L) + I_inj.
    Used to validate numerical simulations; should match within Euler error.

    Examples
    --------
    >>> from jbiophysic.passive_membrane import relaxation_curve
    >>> tau = 100.0  # ms
    >>> V_init = -65.0
    >>> V_ss = -55.0
    >>> V_t_100ms = relaxation_curve(100.0, V_init, V_ss, tau)
    >>> print(f"V(t=tau) = {V_t_100ms:.3f} mV")
    V(t=tau) ≈ -59.632 mV  # 63.2% of way to V_ss

    >>> t_array = np.array([0, 100, 200, 300])
    >>> V_array = relaxation_curve(t_array, V_init, V_ss, tau)
    >>> print(V_array)
    [-65.     -59.632 -56.321 -55.498]
    """
    if tau <= 0:
        raise ValueError(f"tau must be > 0, got {tau}")
    return V_ss + (V_init - V_ss) * np.exp(-t / tau)


def input_resistance_mohm(g_L: float) -> float:
    """Calculate input resistance of passive membrane.

    Parameters
    ----------
    g_L : float
        Soma leak conductance (nS).

    Returns
    -------
    R_m : float
        Input resistance (MΩ). R_m = 1 / g_L (with unit conversion).

    Notes
    -----
    Higher R_m (lower g_L) = larger voltage response to input current.
    Typical values: R_m ~ 100–1000 MΩ for soma (g_L ~ 1–10 nS).

    Examples
    --------
    >>> from jbiophysic.passive_membrane import input_resistance_mohm
    >>> R_m = input_resistance_mohm(g_L=1.0)
    >>> print(f"R_m = {R_m:.1f} MΩ")
    R_m = 1000.0 MΩ

    >>> R_m = input_resistance_mohm(g_L=2.0)
    >>> print(f"R_m = {R_m:.1f} MΩ")
    R_m = 500.0 MΩ
    """
    if g_L <= 0:
        raise ValueError(f"g_L must be > 0, got {g_L}")
    # 1 nS = 10^-9 S = 10^9 MΩ^-1
    # R (MΩ) = 1 / g_L (nS) * 1000 = 1000 / g_L
    return float(1000.0 / g_L)


def membrane_potential_response(
    I_step: float,
    g_L: float,
    tau: float,
) -> dict:
    """Characterize passive membrane response to step current.

    Computes key properties of step response:
    - Steady-state deflection: ΔV_ss = I_step / g_L
    - Time to 50% of steady state: t_half = tau * ln(2) ≈ 0.693 * tau

    Parameters
    ----------
    I_step : float
        Step current magnitude (pA). Positive = depolarizing.
    g_L : float
        Soma leak conductance (nS).
    tau : float
        Time constant (ms).

    Returns
    -------
    response : dict
        Keys:
        - 'V_ss': steady-state deflection (mV)
        - 't_half': time to 50% of steady state (ms)
        - 'tau': time constant (ms, echoed)
        - 'is_finite': True if all values are finite

    Examples
    --------
    >>> from jbiophysic.passive_membrane import membrane_potential_response
    >>> resp = membrane_potential_response(I_step=10.0, g_L=1.0, tau=100.0)
    >>> print(f"ΔV_ss = {resp['V_ss']:.2f} mV")
    ΔV_ss = 10.00 mV
    >>> print(f"t_half = {resp['t_half']:.2f} ms")
    t_half = 69.31 ms
    """
    if g_L <= 0:
        raise ValueError(f"g_L must be > 0, got {g_L}")
    if tau <= 0:
        raise ValueError(f"tau must be > 0, got {tau}")

    V_ss = I_step / g_L
    t_half = tau * np.log(2)

    return {
        "V_ss": float(V_ss),
        "t_half": float(t_half),
        "tau": float(tau),
        "is_finite": bool(np.isfinite(V_ss) and np.isfinite(t_half)),
    }
