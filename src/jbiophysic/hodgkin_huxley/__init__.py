"""Hodgkin-Huxley voltage-gated ion channel kinetics.

Single-compartment HH model: soma with voltage-gated Na, K, and leak conductances.

C_m dV/dt = -g_Na m^3 h (V - E_Na) - g_K n^4 (V - E_K) - g_L (V - E_L) + I_inj

dm/dt = alpha_m(V)(1-m) - beta_m(V)m
dh/dt = alpha_h(V)(1-h) - beta_h(V)h
dn/dt = alpha_n(V)(1-n) - beta_n(V)n

v0.3.0-v0.3.6: Teaching scaffold, truth_safe_unverified, computational_scaffold.
"""

from __future__ import annotations

from .diagnostics import (
    hh_spike_detection,
    hh_stability_report,
    hh_state_check,
)
from .simulator import (
    HodgkinHuxleyParams,
    hh_currents,
    hh_find_rest_voltage,
    hh_rate_functions,
    hh_rhs_current_at_steady_gates,
    hh_simulate,
    hh_steady_state_gates,
    hh_step,
)

__all__ = [
    # Simulator
    "HodgkinHuxleyParams",
    "hh_rate_functions",
    "hh_steady_state_gates",
    "hh_rhs_current_at_steady_gates",
    "hh_find_rest_voltage",
    "hh_currents",
    "hh_step",
    "hh_simulate",
    # Diagnostics
    "hh_state_check",
    "hh_spike_detection",
    "hh_stability_report",
]
