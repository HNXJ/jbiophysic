"""Tests for v0.1 units and numerical discipline chapter."""

import numpy as np

from jbiophysic.units import (
    compare_dtype_passive_membrane,
    conductance_per_soma_area,
    dtype_comparison_report,
    # v0.1.4: Stability diagnostics
    finite_value_check,
    integration_stability_report,
    magnitude_diagnostics,
    monotonic_blow_up_check,
    # v0.1.2: Conversions
    tau_membrane_ms,
)


class TestDtypeComparison:
    """Tests for float32 vs float64 comparison (v0.1.3)."""

    def test_compare_dtype_passive_membrane_runs(self):
        """Test that dtype comparison runs without error."""
        result = compare_dtype_passive_membrane(
            duration_ms=10.0,
            dt_ms=0.1,
            I_inj_pA=5.0,
            seed=42,
        )
        assert result is not None
        assert result.dtype_32 == "float32"
        assert result.dtype_64 == "float64"

    def test_dtype_comparison_finite(self):
        """Test that both dtypes produce finite values."""
        result = compare_dtype_passive_membrane(
            duration_ms=50.0,
            dt_ms=0.1,
            I_inj_pA=10.0,
            seed=42,
        )
        assert result.both_finite_32
        assert result.both_finite_64

    def test_dtype_comparison_error_bounds(self):
        """Test that float32 vs float64 error is small for passive membrane."""
        result = compare_dtype_passive_membrane(
            duration_ms=100.0,
            dt_ms=0.1,
            I_inj_pA=10.0,
            seed=42,
        )
        # For passive membrane, float32 should be within ~0.01 mV of float64
        assert result.max_absolute_error < 0.1

    def test_dtype_comparison_report_json_safe(self):
        """Test that dtype report is JSON-safe (no NaN/Inf)."""
        result = compare_dtype_passive_membrane(
            duration_ms=50.0,
            dt_ms=0.1,
            I_inj_pA=10.0,
            seed=42,
        )
        report = dtype_comparison_report(result)

        # All values must be JSON-serializable
        for key, value in report.items():
            if isinstance(value, (list, tuple)):
                for item in value:
                    assert np.isfinite(item), f"{key}: {item} is not finite"
            elif isinstance(value, (int, float)):
                if isinstance(value, float):
                    assert np.isfinite(value), f"{key}: {value} is not finite"

    def test_dtype_comparison_spike_counts_match(self):
        """Test that spike counts are similar between float32 and float64."""
        result = compare_dtype_passive_membrane(
            duration_ms=100.0,
            dt_ms=0.1,
            I_inj_pA=15.0,
            seed=42,
        )
        # With passive membrane, no spikes; both should be zero
        assert result.spike_count_32 == result.spike_count_64 == 0


class TestStabilityDiagnostics:
    """Tests for stability diagnostics (v0.1.4)."""

    def test_finite_value_check_all_finite(self):
        """Test finite value check with finite array."""
        arr = np.array([1.0, 2.0, 3.0])
        report = finite_value_check(arr, "test")

        assert report["is_finite"]
        assert not report["has_nan"]
        assert not report["has_inf"]
        assert report["n_nan"] == 0
        assert report["n_inf"] == 0

    def test_finite_value_check_with_nan(self):
        """Test finite value check with NaN."""
        arr = np.array([1.0, np.nan, 3.0])
        report = finite_value_check(arr, "test")

        assert not report["is_finite"]
        assert report["has_nan"]
        assert report["n_nan"] == 1

    def test_finite_value_check_with_inf(self):
        """Test finite value check with Inf."""
        arr = np.array([1.0, np.inf, 3.0])
        report = finite_value_check(arr, "test")

        assert not report["is_finite"]
        assert report["has_inf"]
        assert report["n_inf"] == 1

    def test_magnitude_diagnostics_in_range(self):
        """Test magnitude diagnostics with in-range values."""
        arr = np.linspace(-80, -50, 100)
        report = magnitude_diagnostics(
            arr, "voltage", expected_range=(-100, 0)
        )

        assert report["in_range"]
        assert report["n_out_of_range"] == 0

    def test_magnitude_diagnostics_out_of_range(self):
        """Test magnitude diagnostics with out-of-range values."""
        arr = np.array([-100, -50, 50, 100])
        report = magnitude_diagnostics(
            arr, "voltage", expected_range=(-80, 30)
        )

        # -100 < -80 (OOR), -50 in range, 50 > 30 (OOR), 100 > 30 (OOR) = 3 OOR
        assert not report["in_range"]
        assert report["n_out_of_range"] == 3

    def test_monotonic_blow_up_check_stable(self):
        """Test blow-up check with stable (exponential decay)."""
        arr = np.exp(-np.linspace(0, 5, 100))
        report = monotonic_blow_up_check(arr, growth_threshold=2.0)

        assert not report["is_blowing_up"]

    def test_monotonic_blow_up_check_unstable(self):
        """Test blow-up check with unstable (exponential growth)."""
        arr = np.exp(np.linspace(0, 5, 100))
        report = monotonic_blow_up_check(arr, growth_threshold=2.0)

        assert report["is_blowing_up"]

    def test_integration_stability_report_stable_membrane(self):
        """Test stability report for stable passive membrane."""
        n_steps = 100
        voltage = -65 + 5 * np.exp(-np.linspace(0, 5, n_steps))
        current = 10.0 * np.ones(n_steps)

        report = integration_stability_report(voltage, current, 0.1, "test")

        assert report["is_stable"]
        assert report["voltage_finite"]["is_finite"]
        assert report["current_finite"]["is_finite"]

    def test_integration_stability_report_nan_detection(self):
        """Test stability report with NaN in voltage."""
        voltage = np.array([-65.0, -60.0, np.nan, -65.0])
        current = np.array([10.0, 10.0, 10.0, 10.0])

        report = integration_stability_report(voltage, current, 0.1, "test")

        assert not report["is_stable"]
        assert report["voltage_finite"]["has_nan"]

    def test_integration_stability_report_json_safe(self):
        """Test that stability report is fully JSON-safe."""
        voltage = np.sin(np.linspace(0, 10, 100)) * 10 - 65
        current = 10 * np.ones(100)

        report = integration_stability_report(voltage, current, 0.1, "test")

        # Recursively check all numeric values are finite
        def check_json_safe(obj):
            if isinstance(obj, dict):
                for v in obj.values():
                    check_json_safe(v)
            elif isinstance(obj, (list, tuple)):
                for item in obj:
                    check_json_safe(item)
            elif isinstance(obj, float):
                assert np.isfinite(obj), f"Non-finite value: {obj}"

        check_json_safe(report)


class TestWrongUnitsNull:
    """Tests for wrong-units null test behavior (v0.1.6)."""

    def test_correct_units_produces_response(self):
        """Test that correct units produce physiological response."""
        # Simulate with correct current (10 pA)
        n_steps = 100
        v = -65.0
        current = 10.0  # pA (correct)
        g_L = 0.1  # nS
        C_m = 1.0  # pF

        v_trace = [v]
        for _ in range(n_steps):
            I_ion = g_L * (v + 65)
            dv_dt = -(I_ion + current) / C_m
            v = v + 0.1 * dv_dt
            v_trace.append(v)

        v_trace = np.array(v_trace)

        # Voltage should change noticeably
        assert abs(v_trace[-1] - v_trace[0]) > 0.1  # mV change

    def test_too_small_current_produces_no_response(self):
        """Test that too-small current (1e-6 scale) produces no response."""
        # Simulate with absurdly small current (0.00001 pA)
        n_steps = 100
        v = -65.0
        current = 10.0 * 1e-6  # pA (absurdly small)
        g_L = 0.1  # nS
        C_m = 1.0  # pF

        v_trace = [v]
        for _ in range(n_steps):
            I_ion = g_L * (v + 65)
            dv_dt = -(I_ion + current) / C_m
            v = v + 0.1 * dv_dt
            v_trace.append(v)

        v_trace = np.array(v_trace)

        # Voltage should barely change
        assert abs(v_trace[-1] - v_trace[0]) < 0.001  # Negligible mV change

    def test_too_large_current_produces_saturation(self):
        """Test that too-large current (1e6 scale) produces instability."""
        # Simulate with absurdly large current (10 million pA)
        n_steps = 100
        v = -65.0
        current = 10.0 * 1e6  # pA (absurdly large)
        g_L = 0.1  # nS
        C_m = 1.0  # pF

        v_trace = [v]
        for _ in range(n_steps):
            I_ion = g_L * (v + 65)
            dv_dt = -(I_ion + current) / C_m
            v = v + 0.1 * dv_dt
            v_trace.append(v)

        v_trace = np.array(v_trace)

        # Voltage should blow up (run away or saturate)
        voltage_change = abs(v_trace[-1] - v_trace[0])
        assert voltage_change > 100  # Huge change


class TestChapter1Integrity:
    """Integration tests for entire v0.1 chapter."""

    def test_all_diagnostics_json_serializable(self):
        """Test that all chapter outputs are JSON-safe."""
        # Run dtype comparison
        dtype_result = compare_dtype_passive_membrane(
            duration_ms=50.0, dt_ms=0.1, I_inj_pA=10.0, seed=42
        )
        dtype_report = dtype_comparison_report(dtype_result)

        # Run stability analysis
        voltage = np.sin(np.linspace(0, 10, 100)) * 10 - 65
        current = 10 * np.ones(100)
        stability_report = integration_stability_report(
            voltage, current, 0.1, "test"
        )

        # Both must be JSON-serializable
        import json

        json.dumps(dtype_report)  # Should not raise
        json.dumps(stability_report)  # Should not raise

    def test_chapter_imports_complete(self):
        """Test that all chapter functions are importable."""
        from jbiophysic.units import (
            compare_dtype_passive_membrane,
            dtype_comparison_report,
            finite_value_check,
            integration_stability_report,
            magnitude_diagnostics,
            monotonic_blow_up_check,
        )

        # All imports should succeed
        assert callable(tau_membrane_ms)
        assert callable(conductance_per_soma_area)
        assert callable(compare_dtype_passive_membrane)
        assert callable(dtype_comparison_report)
        assert callable(finite_value_check)
        assert callable(magnitude_diagnostics)
        assert callable(monotonic_blow_up_check)
        assert callable(integration_stability_report)
