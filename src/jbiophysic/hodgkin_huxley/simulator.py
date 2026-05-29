"""Hodgkin-Huxley simulator: rate functions, currents, integration.

Canonical parameters from Hodgkin-Huxley squid axon (6.3°C, 1952).
Absolute soma units: C_m in pF, conductances in nS, currents in pA, V in mV, t in ms.

Removable singularities in alpha_m and alpha_n are handled via direct formula
at special voltages (L'Hôpital's rule applied).

Sign Convention (Outward-Positive):
  I_Na = g_Na_bar m^3 h (V - E_Na)  [outward positive]
  I_K  = g_K_bar n^4 (V - E_K)      [outward positive]
  I_L  = g_L (V - E_L)              [outward positive]
  I_ion = I_Na + I_K + I_L          [total ionic current, outward positive]
  I_rhs = I_inj - I_ion             [net RHS current into cell]
  dV/dt = I_rhs / C_m

v0.3.0-v0.3.6
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np


class HodgkinHuxleyParams(NamedTuple):
    """Parameters for single-compartment Hodgkin-Huxley model (absolute soma units).

    Attributes
    ----------
    C_m : float
        Soma membrane capacitance (pF). Typical: 100 pF.
    g_Na_bar : float
        Maximum Na conductance (nS). Typical: 12 nS.
    g_K_bar : float
        Maximum K conductance (nS). Typical: 3.6 nS.
    g_L : float
        Leak conductance (nS). Typical: 0.3 nS.
    E_Na : float
        Na reversal potential (mV). Typical: +60 mV.
    E_K : float
        K reversal potential (mV). Typical: -77 mV.
    E_L : float
        Leak reversal potential (mV). Typical: -54 mV.

    Notes
    -----
    These are canonical values from the original Hodgkin-Huxley squid axon
    (Hodgkin & Huxley, 1952) at 6.3°C. Temperature scaling not included.
    """

    C_m: float
    g_Na_bar: float
    g_K_bar: float
    g_L: float
    E_Na: float
    E_K: float
    E_L: float


def hh_rate_functions(V: float) -> dict:
    """Compute Hodgkin-Huxley rate constants (alpha and beta) at voltage V.

    Handles removable singularities in alpha_m and alpha_n using L'Hôpital's rule.

    Parameters
    ----------
    V : float
        Membrane voltage (mV). Relative to resting state (E_L ≈ -54 mV).

    Returns
    -------
    rates : dict
        Keys: 'alpha_m', 'beta_m', 'alpha_h', 'beta_h', 'alpha_n', 'beta_n'
        All values in 1/ms.

    Notes
    -----
    Standard Hodgkin-Huxley squid axon (6.3°C):

    alpha_m(V) = 0.1(V+40) / (1 - exp(-(V+40)/10))
    At V = -40, numerator = 0, denominator = 0 → use L'Hôpital: α_m = 1.0 (1/ms)

    alpha_n(V) = 0.01(V+55) / (1 - exp(-(V+55)/10))
    At V = -55, numerator = 0, denominator = 0 → use L'Hôpital: α_n = 0.1 (1/ms)

    References
    ----------
    Hodgkin, A. L., & Huxley, A. F. (1952). A quantitative description of
    membrane current and its application to conduction and excitation in nerve.
    J. Physiology, 117(4), 500–544.
    """
    tolerance = 1e-6

    # alpha_m: handle singularity at V = -40 mV
    if abs(V + 40) < tolerance:
        alpha_m = 1.0  # L'Hôpital's rule: lim (0.1 * 10) / (10 * exp(0)) = 1.0
    else:
        alpha_m = 0.1 * (V + 40) / (1.0 - np.exp(-(V + 40) / 10.0))

    beta_m = 4.0 * np.exp(-(V + 65) / 18.0)

    # alpha_h
    alpha_h = 0.07 * np.exp(-(V + 65) / 20.0)

    beta_h = 1.0 / (1.0 + np.exp(-(V + 35) / 10.0))

    # alpha_n: handle singularity at V = -55 mV
    if abs(V + 55) < tolerance:
        alpha_n = 0.1  # L'Hôpital's rule: lim (0.01 * 10) / (10 * exp(0)) = 0.1
    else:
        alpha_n = 0.01 * (V + 55) / (1.0 - np.exp(-(V + 55) / 10.0))

    beta_n = 0.125 * np.exp(-(V + 65) / 80.0)

    return {
        "alpha_m": float(alpha_m),
        "beta_m": float(beta_m),
        "alpha_h": float(alpha_h),
        "beta_h": float(beta_h),
        "alpha_n": float(alpha_n),
        "beta_n": float(beta_n),
    }


def hh_steady_state_gates(V: float) -> tuple:
    """Compute steady-state gating variables at voltage V.

    Parameters
    ----------
    V : float
        Membrane voltage (mV).

    Returns
    -------
    m_inf, h_inf, n_inf : tuple of float
        Steady-state values of m, h, n gates [0, 1].
    """
    rates = hh_rate_functions(V)
    m_inf = rates["alpha_m"] / (rates["alpha_m"] + rates["beta_m"])
    h_inf = rates["alpha_h"] / (rates["alpha_h"] + rates["beta_h"])
    n_inf = rates["alpha_n"] / (rates["alpha_n"] + rates["beta_n"])
    return float(m_inf), float(h_inf), float(n_inf)


def hh_currents(
    V: float,
    m: float,
    h: float,
    n: float,
    params: HodgkinHuxleyParams,
    I_inj: float = 0.0,
) -> dict:
    """Compute ionic currents at given voltage and gate states.

    Sign convention: outward-positive (depolarizing current is negative).

    Parameters
    ----------
    V : float
        Membrane voltage (mV).
    m : float
        Na activation gate [0, 1].
    h : float
        Na inactivation gate [0, 1].
    n : float
        K activation gate [0, 1].
    params : HodgkinHuxleyParams
        Channel parameters.
    I_inj : float, optional
        Injected current (pA). Default 0.

    Returns
    -------
    currents : dict
        Keys: 'I_Na', 'I_K', 'I_L', 'I_ion', 'I_rhs'
        All in pA.
        - I_Na, I_K, I_L: individual ionic currents (outward positive)
        - I_ion: total ionic current = I_Na + I_K + I_L (outward positive)
        - I_rhs: net RHS current = I_inj - I_ion (into cell positive)
    """
    I_Na = params.g_Na_bar * (m**3) * h * (V - params.E_Na)
    I_K = params.g_K_bar * (n**4) * (V - params.E_K)
    I_L = params.g_L * (V - params.E_L)

    I_ion = I_Na + I_K + I_L
    I_rhs = I_inj - I_ion

    return {
        "I_Na": float(I_Na),
        "I_K": float(I_K),
        "I_L": float(I_L),
        "I_ion": float(I_ion),
        "I_rhs": float(I_rhs),
    }


def hh_rhs_current_at_steady_gates(V: float, params: HodgkinHuxleyParams) -> float:
    """Compute I_rhs at voltage V with steady-state gates and no injection.

    Parameters
    ----------
    V : float
        Membrane voltage (mV).
    params : HodgkinHuxleyParams
        Channel parameters.

    Returns
    -------
    I_rhs : float
        Right-hand-side current (pA) = I_inj - I_ion at zero injection.
    """
    m_ss, h_ss, n_ss = hh_steady_state_gates(V)
    currents = hh_currents(V, m_ss, h_ss, n_ss, params, I_inj=0.0)
    return currents["I_rhs"]


def hh_find_rest_voltage(
    params: HodgkinHuxleyParams,
    v_min: float = -90.0,
    v_max: float = -40.0,
    n_grid: int = 2000,
) -> float:
    """Find the voltage at which I_rhs = 0 with steady-state gates and no injection.

    This is the equilibrium voltage where the membrane remains stable.

    Parameters
    ----------
    params : HodgkinHuxleyParams
        Channel parameters.
    v_min : float, optional
        Lower voltage bound for search (mV). Default -90.0.
    v_max : float, optional
        Upper voltage bound for search (mV). Default -40.0.
    n_grid : int, optional
        Number of grid points for search. Default 2000.

    Returns
    -------
    V_rest : float
        Voltage at which I_rhs ≈ 0 (equilibrium voltage, mV).
    """
    voltages = np.linspace(v_min, v_max, n_grid)
    rhs_currents = np.array(
        [hh_rhs_current_at_steady_gates(float(v), params) for v in voltages]
    )
    idx = int(np.argmin(np.abs(rhs_currents)))
    return float(voltages[idx])


def hh_step(
    V: float,
    m: float,
    h: float,
    n: float,
    params: HodgkinHuxleyParams,
    I_inj: float,
    dt_ms: float,
) -> tuple:
    """Single forward-Euler step for Hodgkin-Huxley model.

    Parameters
    ----------
    V, m, h, n : float
        Current state variables.
    params : HodgkinHuxleyParams
        Channel parameters.
    I_inj : float
        Injected current (pA).
    dt_ms : float
        Timestep (ms).

    Returns
    -------
    (V_new, m_new, h_new, n_new) : tuple of float
        Updated state after one step.
    """
    # Compute rate constants at current V
    rates = hh_rate_functions(V)

    # Compute currents
    currents = hh_currents(V, m, h, n, params, I_inj)

    # Update V: C_m dV/dt = I_rhs = I_inj - I_ion
    dV_dt = currents["I_rhs"] / params.C_m
    V_new = V + dt_ms * dV_dt

    # Update gates: dx/dt = alpha(V)(1-x) - beta(V)x
    dm_dt = rates["alpha_m"] * (1.0 - m) - rates["beta_m"] * m
    m_new = m + dt_ms * dm_dt

    dh_dt = rates["alpha_h"] * (1.0 - h) - rates["beta_h"] * h
    h_new = h + dt_ms * dh_dt

    dn_dt = rates["alpha_n"] * (1.0 - n) - rates["beta_n"] * n
    n_new = n + dt_ms * dn_dt

    return (float(V_new), float(m_new), float(h_new), float(n_new))


def hh_simulate(
    V_init: float,
    m_init: float,
    h_init: float,
    n_init: float,
    params: HodgkinHuxleyParams,
    I_inj: float,
    dt_ms: float,
    duration_ms: float,
) -> dict:
    """Simulate Hodgkin-Huxley model.

    Parameters
    ----------
    V_init, m_init, h_init, n_init : float
        Initial state values. Typically: V_init ≈ -65 mV; gates ≈ steady-state at V_init.
    params : HodgkinHuxleyParams
        Channel parameters.
    I_inj : float
        Constant injected current (pA).
    dt_ms : float
        Integration timestep (ms). Recommend dt < 0.1 ms for stability.
    duration_ms : float
        Simulation duration (ms).

    Returns
    -------
    result : dict
        Keys:
        - 'time': array of timestamps (ms)
        - 'V': voltage trace (mV)
        - 'm', 'h', 'n': gating variable traces
        - 'I_Na', 'I_K', 'I_L': individual current traces (pA)
        - 'I_ion': total ionic current (pA, outward positive)
        - 'I_rhs': net RHS current (pA, into cell positive)
        - 'all_finite': bool

    Notes
    -----
    Uses forward Euler; stable for dt < 0.1 * tau where tau ~ 1 ms.
    """
    n_steps = int(round(duration_ms / dt_ms))

    # Initialize traces
    time = np.linspace(0, duration_ms, n_steps + 1)
    V_trace = [V_init]
    m_trace = [m_init]
    h_trace = [h_init]
    n_trace = [n_init]
    I_Na_trace = []
    I_K_trace = []
    I_L_trace = []
    I_ion_trace = []
    I_rhs_trace = []

    # Initial currents
    currents = hh_currents(V_init, m_init, h_init, n_init, params, I_inj)
    I_Na_trace.append(currents["I_Na"])
    I_K_trace.append(currents["I_K"])
    I_L_trace.append(currents["I_L"])
    I_ion_trace.append(currents["I_ion"])
    I_rhs_trace.append(currents["I_rhs"])

    # Integration
    V, m, h, n = V_init, m_init, h_init, n_init
    for _ in range(n_steps):
        V, m, h, n = hh_step(V, m, h, n, params, I_inj, dt_ms)
        V_trace.append(V)
        m_trace.append(m)
        h_trace.append(h)
        n_trace.append(n)

        currents = hh_currents(V, m, h, n, params, I_inj)
        I_Na_trace.append(currents["I_Na"])
        I_K_trace.append(currents["I_K"])
        I_L_trace.append(currents["I_L"])
        I_ion_trace.append(currents["I_ion"])
        I_rhs_trace.append(currents["I_rhs"])

    # Convert to arrays
    V_array = np.array(V_trace, dtype=np.float64)
    m_array = np.array(m_trace, dtype=np.float64)
    h_array = np.array(h_trace, dtype=np.float64)
    n_array = np.array(n_trace, dtype=np.float64)
    I_Na_array = np.array(I_Na_trace, dtype=np.float64)
    I_K_array = np.array(I_K_trace, dtype=np.float64)
    I_L_array = np.array(I_L_trace, dtype=np.float64)
    I_ion_array = np.array(I_ion_trace, dtype=np.float64)
    I_rhs_array = np.array(I_rhs_trace, dtype=np.float64)

    all_finite = bool(
        np.all(np.isfinite(V_array))
        and np.all(np.isfinite(m_array))
        and np.all(np.isfinite(h_array))
        and np.all(np.isfinite(n_array))
        and np.all(np.isfinite(I_ion_array))
        and np.all(np.isfinite(I_rhs_array))
    )

    return {
        "time": time,
        "V": V_array,
        "m": m_array,
        "h": h_array,
        "n": n_array,
        "I_Na": I_Na_array,
        "I_K": I_K_array,
        "I_L": I_L_array,
        "I_ion": I_ion_array,
        "I_rhs": I_rhs_array,
        "all_finite": all_finite,
    }
