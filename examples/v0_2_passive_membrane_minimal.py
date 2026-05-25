"""v0.2.2–v0.2.6 example: Passive membrane dynamics and stability.

This example demonstrates:
1. v0.2.2: Minimal simulator with Euler integration
2. v0.2.3: Steady-state voltage and time constant diagnostics
3. v0.2.5: Parametric sweep (C_m, g_L, I_inj effects)
4. v0.2.6: Timestep stability null test

All units are biophysical SI: mV, ms, pA, mS/cm², μF/cm².
truth_mode: truth_safe_unverified, computational_scaffold
"""

from __future__ import annotations

import numpy as np

from jbiophysic.passive_membrane import (
    PassiveMembraneParams,
    passive_membrane_simulate,
    tau_membrane_ms,
    steady_state_voltage,
    relaxation_curve,
)
from jbiophysic.units import monotonic_blow_up_check


def example_v0_2_2_minimal_simulator():
    """v0.2.2: Minimal passive membrane simulator."""
    print("\n" + "=" * 80)
    print("v0.2.2: Minimal Passive Membrane Simulator")
    print("=" * 80)

    # Define parameters
    params = PassiveMembraneParams(C_m=1.0, g_L=0.1, E_L=-65.0)

    print("\nParameters:")
    print(f"  C_m = {params.C_m} μF/cm² (membrane capacitance)")
    print(f"  g_L = {params.g_L} mS/cm² (leak conductance)")
    print(f"  E_L = {params.E_L} mV (leak reversal potential)")

    # Calculate derived properties
    tau = tau_membrane_ms(params.C_m, params.g_L)
    print(f"\nDerived properties:")
    print(f"  tau = C_m / g_L = {tau:.1f} ms (time constant)")

    # Simulate at rest
    print("\n" + "-" * 80)
    print("Scenario 1: At rest (I_inj = 0)")
    print("-" * 80)
    V_trace_rest = passive_membrane_simulate(
        V_init=-65.0,
        params=params,
        I_inj=0.0,
        dt_ms=0.1,
        duration_ms=100.0,
    )
    print(f"V(t=0) = {V_trace_rest[0]:.2f} mV")
    print(f"V(t=100 ms) = {V_trace_rest[-1]:.2f} mV")
    print(f"Expected: V_rest = E_L = {params.E_L} mV")
    print(f"Result: STABLE ✓" if np.allclose(V_trace_rest, params.E_L) else "Result: FAILED ✗")

    # Simulate with step current
    print("\n" + "-" * 80)
    print("Scenario 2: Step current (I_inj = 10 pA)")
    print("-" * 80)
    I_inj = 10.0
    V_ss_expected = steady_state_voltage(params.E_L, params.g_L, I_inj)

    V_trace_step = passive_membrane_simulate(
        V_init=-65.0,
        params=params,
        I_inj=I_inj,
        dt_ms=0.01,  # Small timestep for accuracy
        duration_ms=200.0,
    )
    print(f"I_inj = {I_inj} pA")
    print(f"Expected V_ss = E_L - I_inj / g_L = {V_ss_expected:.3f} mV")
    print(f"Simulated V(t=200 ms) = {V_trace_step[-1]:.3f} mV")
    print(f"Error = {abs(V_trace_step[-1] - V_ss_expected):.5f} mV")

    # Compare with exact solution
    print("\n" + "-" * 80)
    print("Scenario 3: Numerical vs. Analytical Solution")
    print("-" * 80)
    V_init = -65.0
    V_ss = V_ss_expected
    t_values = np.array([0, tau, 2*tau, 5*tau])

    print(f"Time (ms)  | Analytical (mV) | Numerical (mV) | Error (mV)")
    print(f"-" * 60)
    for t in t_values:
        idx = int(t / 0.01)
        if idx < len(V_trace_step):
            V_analytical = relaxation_curve(t, V_init, V_ss, tau)
            V_numerical = V_trace_step[idx]
            error = abs(V_analytical - V_numerical)
            print(
                f"{t:10.2f} | {V_analytical:15.6f} | {V_numerical:14.6f} | {error:9.6f}"
            )


def example_v0_2_5_parametric_sweep():
    """v0.2.5: Sweep over membrane parameters."""
    print("\n" + "=" * 80)
    print("v0.2.5: Parametric Sweep (C_m, g_L, I_inj)")
    print("=" * 80)

    # Test different C_m values
    print("\nEffect of C_m (membrane capacitance):")
    print(f"{'C_m':>6} | {'tau':>8} | {'V_ss':>8} | {'V(200ms)':>10}")
    print("-" * 40)

    C_m_values = [0.5, 1.0, 2.0]
    g_L = 0.1
    I_inj = 10.0

    for C_m in C_m_values:
        params = PassiveMembraneParams(C_m=C_m, g_L=g_L, E_L=-65.0)
        tau = tau_membrane_ms(C_m, g_L)
        V_ss = steady_state_voltage(-65.0, g_L, I_inj)

        V_trace = passive_membrane_simulate(
            V_init=-65.0,
            params=params,
            I_inj=I_inj,
            dt_ms=0.05,
            duration_ms=200.0,
        )

        print(f"{C_m:6.1f} | {tau:8.1f} | {V_ss:8.3f} | {V_trace[-1]:10.3f}")

    # Test different g_L values
    print("\nEffect of g_L (leak conductance):")
    print(f"{'g_L':>6} | {'tau':>8} | {'V_ss':>8} | {'V(200ms)':>10}")
    print("-" * 40)

    g_L_values = [0.05, 0.1, 0.2]
    C_m = 1.0

    for g_L in g_L_values:
        params = PassiveMembraneParams(C_m=C_m, g_L=g_L, E_L=-65.0)
        tau = tau_membrane_ms(C_m, g_L)
        V_ss = steady_state_voltage(-65.0, g_L, I_inj)

        V_trace = passive_membrane_simulate(
            V_init=-65.0,
            params=params,
            I_inj=I_inj,
            dt_ms=0.05,
            duration_ms=200.0,
        )

        print(f"{g_L:6.2f} | {tau:8.1f} | {V_ss:8.3f} | {V_trace[-1]:10.3f}")

    # Test different I_inj values
    print("\nEffect of I_inj (injected current):")
    print(f"{'I_inj':>8} | {'V_ss':>8} | {'V(200ms)':>10}")
    print("-" * 35)

    I_inj_values = [0.0, 10.0, 50.0, 100.0]
    params = PassiveMembraneParams(C_m=1.0, g_L=0.1, E_L=-65.0)

    for I_inj in I_inj_values:
        V_ss = steady_state_voltage(params.E_L, params.g_L, I_inj)
        V_trace = passive_membrane_simulate(
            V_init=-65.0,
            params=params,
            I_inj=I_inj,
            dt_ms=0.05,
            duration_ms=200.0,
        )
        print(f"{I_inj:8.1f} | {V_ss:8.3f} | {V_trace[-1]:10.3f}")


def example_v0_2_6_timestep_stability_null():
    """v0.2.6: Null test for timestep stability."""
    print("\n" + "=" * 80)
    print("v0.2.6: Timestep Stability Null Test")
    print("=" * 80)

    print("\nHypothesis: Forward Euler is stable for dt < 2*tau, unstable for dt > 2*tau")

    params = PassiveMembraneParams(C_m=1.0, g_L=0.1, E_L=-65.0)
    tau = tau_membrane_ms(params.C_m, params.g_L)

    print(f"\nMemory parameters: C_m={params.C_m} μF/cm², g_L={params.g_L} mS/cm²")
    print(f"Time constant: tau = {tau:.1f} ms")
    print(f"Stability threshold: dt < 2*tau = {2*tau:.1f} ms\n")

    dt_values = np.array([0.5*tau, 1.0*tau, 1.5*tau, 1.9*tau, 2.0*tau, 2.5*tau, 3.0*tau])

    print(f"{'dt (ms)':>10} | {'dt/tau':>8} | {'Stable?':>10} | {'V_max':>10} | {'Has NaN?':>10}")
    print("-" * 65)

    for dt_ms in dt_values:
        try:
            V_trace = passive_membrane_simulate(
                V_init=-65.0,
                params=params,
                I_inj=10.0,
                dt_ms=dt_ms,
                duration_ms=50.0,
            )

            has_nan = np.any(np.isnan(V_trace))
            V_max = np.nanmax(np.abs(V_trace)) if not has_nan else np.inf

            # Check for blow-up using v0.1 diagnostic
            blow_up_check = monotonic_blow_up_check(
                np.abs(V_trace), growth_threshold=2.0, name=f"dt={dt_ms}"
            )
            is_stable = not (blow_up_check["is_blowing_up"] or has_nan or V_max > 500)

            status = "STABLE ✓" if is_stable else "UNSTABLE ✗"

            print(
                f"{dt_ms:10.3f} | {dt_ms/tau:8.2f} | {status:>10} | {V_max:10.2f} | {str(has_nan):>10}"
            )
        except Exception as e:
            print(f"{dt_ms:10.3f} | {dt_ms/tau:8.2f} | ERROR | {str(e)[:20]}")

    print("\n" + "=" * 80)
    print("Interpretation:")
    print("=" * 80)
    print("✓ STABLE (dt < 2*tau): Voltage relaxes smoothly; no blow-up")
    print("✗ UNSTABLE (dt > 2*tau): Voltage diverges or produces NaN")
    print("\nLesson: Explicit Euler requires dt < 2*tau for passive membrane.")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    example_v0_2_2_minimal_simulator()
    example_v0_2_5_parametric_sweep()
    example_v0_2_6_timestep_stability_null()
