"""v0.1.6 example: Wrong-units null test (negative control).

This example demonstrates what happens when units are incorrect or miscalibrated.
It's a teaching tool, not a package failure.
"""

from __future__ import annotations

import numpy as np

from jbiophysic.units import integration_stability_report


def simulate_passive_membrane_with_scale(
    duration_ms: float,
    dt_ms: float,
    current_scale: float,  # Multiplier on current (1.0 = correct, 1e-6 = wrong)
    name: str = "simulation",
) -> dict:
    """
    Simulate passive membrane with scaled current (simulating unit errors).

    Parameters
    ----------
    duration_ms : float
        Duration (ms)
    dt_ms : float
        Timestep (ms)
    current_scale : float
        Scale factor for injected current
        1.0 = physically reasonable (10 pA)
        1e-6 = absurdly small (0.00001 pA = wrong units)
    name : str
        Simulation name

    Returns
    -------
    report : dict
        Stability and outcome report
    """
    n_steps = int(round(duration_ms / dt_ms))

    v = -65.0  # mV
    g_L = 0.1  # nS
    E_L = -65.0  # mV
    C_m = 1.0  # pF
    I_inj = 10.0 * current_scale  # Scaled current

    v_trace = [v]

    for _ in range(n_steps):
        I_ion = g_L * (v - E_L)
        dv_dt = -(I_ion + I_inj) / C_m
        v = v + dt_ms * dv_dt
        v_trace.append(v)

    v_trace = np.array(v_trace, dtype=np.float64)
    current_trace = np.ones_like(v_trace) * I_inj

    diag = integration_stability_report(v_trace, current_trace, dt_ms, name=name)

    return {
        "name": name,
        "current_scale": current_scale,
        "voltage_trace": v_trace,
        "current_trace": current_trace,
        "diagnostics": diag,
        "terminal_voltage_mV": float(v_trace[-1]),
        "voltage_change_mV": float(v_trace[-1] - v_trace[0]),
    }


def run_wrong_units_null_test():
    """Run wrong-units null test example."""
    print("\n" + "=" * 80)
    print("v0.1.6: Wrong-Units Null Test (Negative Control)")
    print("=" * 80)

    print("\nHypothesis: Incorrect unit scaling produces obviously wrong results.")
    print("This is a teaching null, not a package bug.\n")

    # Correct units
    correct = simulate_passive_membrane_with_scale(
        duration_ms=100.0,
        dt_ms=0.1,
        current_scale=1.0,  # Correct: 10 pA
        name="correct_units",
    )

    # Wrong units (current 1e-6 too small)
    wrong = simulate_passive_membrane_with_scale(
        duration_ms=100.0,
        dt_ms=0.1,
        current_scale=1e-6,  # Wrong: 0.00001 pA (absurdly small)
        name="wrong_units_1e6_too_small",
    )

    # Extremely wrong (current 1e6 too large)
    extremely_wrong = simulate_passive_membrane_with_scale(
        duration_ms=100.0,
        dt_ms=0.1,
        current_scale=1e6,  # Extremely wrong: 10 million pA
        name="wrong_units_1e6_too_large",
    )

    print("=" * 80)
    print("CORRECT (current_scale=1.0, I_inj=10 pA):")
    print("=" * 80)
    print(f"Terminal voltage: {correct['terminal_voltage_mV']:.2f} mV")
    print(f"Voltage change: {correct['voltage_change_mV']:.3f} mV")
    print(f"Stable: {correct['diagnostics']['is_stable']}")
    print(f"Blowing up: {correct['diagnostics']['voltage_blow_up']['is_blowing_up']}")

    print("\n" + "=" * 80)
    print("WRONG #1 (current_scale=1e-6, I_inj=0.00001 pA):")
    print("=" * 80)
    print(f"Terminal voltage: {wrong['terminal_voltage_mV']:.2f} mV")
    print(f"Voltage change: {wrong['voltage_change_mV']:.3f} mV")
    print(f"Stable: {wrong['diagnostics']['is_stable']}")
    print(f"Blowing up: {wrong['diagnostics']['voltage_blow_up']['is_blowing_up']}")
    print("→ No response to stimulus: current is negligibly small!")

    print("\n" + "=" * 80)
    print("WRONG #2 (current_scale=1e6, I_inj=10 million pA):")
    print("=" * 80)
    print(f"Terminal voltage: {extremely_wrong['terminal_voltage_mV']:.2f} mV")
    print(f"Voltage change: {extremely_wrong['voltage_change_mV']:.3f} mV")
    print(f"Stable: {extremely_wrong['diagnostics']['is_stable']}")
    print(f"Blowing up: {extremely_wrong['diagnostics']['voltage_blow_up']['is_blowing_up']}")
    print("→ Catastrophic instability or voltage saturation: current is absurdly large!")

    print("\n" + "=" * 80)
    print("Lesson:")
    print("=" * 80)
    print("Unit errors are NOT subtle. They produce obviously broken behavior:")
    print("  - Too small: no response (silent neuron)")
    print("  - Too large: explosion or saturation")
    print("  - Correct: reasonable voltage deflection")
    print("\nNumerical stability is necessary but not sufficient for claims.")
    print("Reasonable behavior under correct units is also necessary.")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    run_wrong_units_null_test()
