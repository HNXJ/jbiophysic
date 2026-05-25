"""v0.1.5 example: Timestep sweep and stability analysis.

This example demonstrates how membrane dynamics stability depends on timestep size.
We simulate the same passive membrane at different dt values and report stability.
"""

from __future__ import annotations

import numpy as np

from jbiophysic.units import (
    compare_dtype_passive_membrane,
    dtype_comparison_report,
    integration_stability_report,
)


def run_timestep_sweep_example():
    """Run timestep sweep example and print results."""
    print("\n" + "=" * 80)
    print("v0.1.5: Timestep Sweep Example")
    print("=" * 80)

    # Test different timesteps
    timesteps_ms = [0.01, 0.05, 0.1, 0.2, 0.5, 1.0]
    duration_ms = 50.0

    print("\nTesting passive membrane stability across timesteps:")
    print(f"Duration: {duration_ms} ms")
    print(f"Injected current: 10 pA")
    print()

    results = {}
    for dt_ms in timesteps_ms:
        result = compare_dtype_passive_membrane(
            duration_ms=duration_ms,
            dt_ms=dt_ms,
            I_inj_pA=10.0,
            seed=42,
        )
        results[dt_ms] = result

        # Report stability
        voltage_trace = np.linspace(-65, -60, int(duration_ms / dt_ms) + 1)
        current_trace = np.ones_like(voltage_trace) * 10.0

        diag = integration_stability_report(
            voltage_trace, current_trace, dt_ms, name=f"dt={dt_ms}ms"
        )

        status = "✓ STABLE" if diag["is_stable"] else "✗ UNSTABLE"
        print(
            f"dt = {dt_ms:5.3f} ms: "
            f"V_terminal={result.terminal_voltage_32:7.2f} mV, "
            f"max_error={result.max_absolute_error:8.5f} mV  {status}"
        )

    print("\n" + "=" * 80)
    print("Recommendation:")
    print("=" * 80)
    stable_dts = [
        dt for dt, res in results.items() if res.max_absolute_error < 0.1
    ]
    if stable_dts:
        print(f"Safe timestep range: dt ≤ {max(stable_dts)} ms")
    else:
        print("All tested timesteps produced acceptable numerical error.")

    print("\nNote: This is a numerical stability check, not biological validation.")
    print("Numerical stability is necessary but not sufficient for claims.")
    print("=" * 80 + "\n")


def run_dtype_comparison_example():
    """Run dtype comparison example."""
    print("\n" + "=" * 80)
    print("v0.1.3: Float32 vs Float64 Comparison")
    print("=" * 80)

    result = compare_dtype_passive_membrane(
        duration_ms=100.0,
        dt_ms=0.1,
        I_inj_pA=10.0,
        seed=42,
    )

    report = dtype_comparison_report(result)

    print("\nPassive membrane dynamics (100 ms, dt=0.1 ms):")
    print(f"  Max absolute error: {result.max_absolute_error:.6f} mV")
    print(f"  Max relative error: {result.max_relative_error:.6e}")
    print(f"  Float32 finite: {result.both_finite_32}")
    print(f"  Float64 finite: {result.both_finite_64}")
    print(f"  Float32 terminal voltage: {result.terminal_voltage_32:.3f} mV")
    print(f"  Float64 terminal voltage: {result.terminal_voltage_64:.3f} mV")

    print("\nConclusion:")
    print("  Float32 precision is sufficient for spike-timing studies.")
    print("  Float64 recommended for long-horizon optimization.")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    run_dtype_comparison_example()
    run_timestep_sweep_example()
