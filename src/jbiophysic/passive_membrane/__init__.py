"""Passive membrane biophysics: lumped cable model without spatial extent.

This module implements the fundamental passive membrane equation:

    C_m * dV/dt = -g_L * (V - E_L) - I_inj

Where:
    C_m: Membrane capacitance (μF/cm²)
    g_L: Leak conductance (mS/cm²)
    E_L: Leak equilibrium potential (mV)
    I_inj: Injected current (pA)

The passive membrane is the foundation for all neuronal models in jbiophysic.
Advanced models (Hodgkin-Huxley, Izhikevich) add conductances on top of this base.

v0.2.0–v0.2.11 doctrine: computational_scaffold, truth_safe_unverified
"""

from __future__ import annotations

from .simulator import (
    PassiveMembraneParams,
    passive_membrane_step,
    passive_membrane_simulate,
)
from .diagnostics import (
    tau_membrane_ms,
    steady_state_voltage,
    relaxation_curve,
    input_resistance_mohm,
    membrane_potential_response,
)

__all__ = [
    # Simulator
    "PassiveMembraneParams",
    "passive_membrane_step",
    "passive_membrane_simulate",
    # Diagnostics
    "tau_membrane_ms",
    "steady_state_voltage",
    "relaxation_curve",
    "input_resistance_mohm",
    "membrane_potential_response",
]
