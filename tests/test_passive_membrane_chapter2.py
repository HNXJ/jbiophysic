"""Tests for v0.2 passive membrane physics chapter.

Validates passive membrane equation, simulator, and diagnostics.
Tests v0.2.2 (executable) and v0.2.3 (diagnostics).

v0.2.2–v0.2.3 gate: All tests PASS before advancing to v0.2.4.
"""

import pytest
import numpy as np

from jbiophysic.passive_membrane import (
    PassiveMembraneParams,
    passive_membrane_step,
    passive_membrane_simulate,
    tau_membrane_ms,
    steady_state_voltage,
    relaxation_curve,
    input_resistance_mohm,
    membrane_potential_response,
)
from jbiophysic.units import (
    finite_value_check,
    magnitude_diagnostics,
    monotonic_blow_up_check,
    integration_stability_report,
)


class TestPassiveMembraneParams:
    """Tests for PassiveMembraneParams NamedTuple."""

    def test_params_creation(self):
        """Test basic parameter creation."""
        params = PassiveMembraneParams(C_m=100.0, g_L=1.0, E_L=-65.0)
        assert params.C_m == 100.0
        assert params.g_L == 1.0
        assert params.E_L == -65.0

    def test_params_json_serializable(self):
        """Test that params can be converted to dict for JSON."""
        params = PassiveMembraneParams(C_m=100.0, g_L=1.0, E_L=-65.0)
        params_dict = params._asdict()
        assert isinstance(params_dict, dict)
        assert all(isinstance(v, (int, float)) for v in params_dict.values())


class TestPassiveMembraneStep:
    """Tests for single-step integration."""

    def test_step_basic(self):
        """Test basic forward Euler step."""
        V = -65.0
        V_new = passive_membrane_step(
            V, C_m=100.0, g_L=1.0, E_L=-65.0, I_inj=0.0, dt_ms=1.0
        )
        # At rest (I_inj=0, V=E_L), dV/dt = 0, so V_new = V
        assert np.isclose(V_new, V)

    def test_step_with_depolarizing_current(self):
        """Test step with positive (depolarizing) current."""
        V = -65.0
        V_new = passive_membrane_step(
            V, C_m=100.0, g_L=1.0, E_L=-65.0, I_inj=10.0, dt_ms=1.0
        )
        # I_inj > 0 should cause dV/dt > 0 (depolarization)
        # dV/dt = (-1.0*(-65-(-65)) + 10) / 100.0 = 10 / 100.0 = 0.1 mV/ms
        # V_new = -65 + 1.0 * 0.1 = -64.9 mV (depolarized)
        assert V_new > V  # Positive current increases voltage

    def test_step_approaches_steady_state(self):
        """Test that repeated stepping approaches steady state."""
        V = -65.0
        params = PassiveMembraneParams(C_m=100.0, g_L=1.0, E_L=-65.0)
        I_inj = 10.0
        V_ss = steady_state_voltage(params.E_L, params.g_L, I_inj)

        # Take many small steps (total 500 ms = 5 * tau)
        for _ in range(5000):
            V = passive_membrane_step(
                V, C_m=params.C_m, g_L=params.g_L, E_L=params.E_L,
                I_inj=I_inj, dt_ms=0.1
            )

        # After 5*tau, should be >99% to steady state
        assert np.isclose(V, V_ss, atol=0.1)

    def test_step_with_zero_current(self):
        """Test that V remains at rest if V=E_L and I_inj=0."""
        V = -65.0
        V_new = passive_membrane_step(
            V, C_m=100.0, g_L=1.0, E_L=-65.0, I_inj=0.0, dt_ms=1.0
        )
        assert np.isclose(V_new, V)

    def test_step_stability_criterion(self):
        """Test that dt < 2*tau is stable."""
        V = -65.0
        C_m = 100.0
        g_L = 1.0
        tau = tau_membrane_ms(C_m, g_L)
        dt_stable = 50.0  # 0.5 * tau

        # Repeated stable stepping shouldn't blow up
        for _ in range(100):
            V = passive_membrane_step(
                V, C_m=C_m, g_L=g_L, E_L=-65.0, I_inj=10.0, dt_ms=dt_stable
            )
            assert np.isfinite(V)


class TestPassiveMembraneSimulate:
    """Tests for full simulation."""

    def test_simulate_basic(self):
        """Test basic simulation run."""
        params = PassiveMembraneParams(C_m=100.0, g_L=1.0, E_L=-65.0)
        V_trace = passive_membrane_simulate(
            V_init=-65.0, params=params, I_inj=0.0, dt_ms=1.0, duration_ms=100.0
        )
        assert len(V_trace) == 101  # 100 ms / 1 ms + 1 initial
        assert V_trace[0] == -65.0  # Initial condition

    def test_simulate_at_rest(self):
        """Test that V remains at rest when I_inj=0 and V=E_L."""
        params = PassiveMembraneParams(C_m=100.0, g_L=1.0, E_L=-65.0)
        V_trace = passive_membrane_simulate(
            V_init=-65.0, params=params, I_inj=0.0, dt_ms=1.0, duration_ms=100.0
        )
        # All values should be -65.0
        assert np.allclose(V_trace, -65.0, atol=1e-6)

    def test_simulate_exponential_relaxation(self):
        """Test that simulation matches exact exponential solution."""
        params = PassiveMembraneParams(C_m=100.0, g_L=1.0, E_L=-65.0)
        I_inj = 10.0
        V_init = -65.0
        V_ss = steady_state_voltage(params.E_L, params.g_L, I_inj)
        tau = tau_membrane_ms(params.C_m, params.g_L)

        V_trace = passive_membrane_simulate(
            V_init=V_init, params=params, I_inj=I_inj, dt_ms=1.0, duration_ms=1000.0
        )

        # Compare against exact solution at t=tau (63.2% of way to V_ss)
        idx_tau = int(tau / 1.0)
        V_exact_tau = relaxation_curve(tau, V_init, V_ss, tau)
        V_numerical_tau = V_trace[idx_tau]

        # Allow 0.1 mV error due to Euler discretization
        assert np.isclose(V_numerical_tau, V_exact_tau, atol=0.1)

    def test_simulate_steady_state_accuracy(self):
        """Test that long simulation reaches correct steady state."""
        params = PassiveMembraneParams(C_m=100.0, g_L=1.0, E_L=-65.0)
        I_inj = 25.0
        V_ss = steady_state_voltage(params.E_L, params.g_L, I_inj)

        V_trace = passive_membrane_simulate(
            V_init=-65.0, params=params, I_inj=I_inj, dt_ms=1.0, duration_ms=5000.0
        )

        # After 5000 ms (>> tau), should reach steady state
        assert np.isclose(V_trace[-1], V_ss, atol=0.01)

    def test_simulate_finite_values(self):
        """Test that all simulated values are finite."""
        params = PassiveMembraneParams(C_m=100.0, g_L=1.0, E_L=-65.0)
        V_trace = passive_membrane_simulate(
            V_init=-65.0, params=params, I_inj=50.0, dt_ms=1.0, duration_ms=200.0
        )
        assert np.all(np.isfinite(V_trace))

    def test_simulate_time_varying_current(self):
        """Test simulation with time-varying injected current."""
        params = PassiveMembraneParams(C_m=100.0, g_L=1.0, E_L=-65.0)
        n_steps = 200
        I_inj_trace = np.linspace(0, 100, n_steps)  # Ramp from 0 to 100 pA

        V_trace = passive_membrane_simulate(
            V_init=-65.0,
            params=params,
            I_inj=I_inj_trace,
            dt_ms=1.0,
            duration_ms=200.0,
        )

        assert len(V_trace) == n_steps + 1
        # With increasing current, voltage should generally increase (depolarize)
        assert V_trace[-1] > V_trace[0]

    def test_simulate_mismatched_current_length(self):
        """Test error handling for mismatched I_inj length."""
        params = PassiveMembraneParams(C_m=100.0, g_L=1.0, E_L=-65.0)
        I_inj_trace = np.array([10.0, 20.0])  # Too short

        with pytest.raises(ValueError):
            passive_membrane_simulate(
                V_init=-65.0,
                params=params,
                I_inj=I_inj_trace,
                dt_ms=1.0,
                duration_ms=200.0,
            )

    def test_simulate_stability_diagnostic(self):
        """Test integration stability via v0.1 diagnostics."""
        params = PassiveMembraneParams(C_m=100.0, g_L=1.0, E_L=-65.0)
        V_trace = passive_membrane_simulate(
            V_init=-65.0, params=params, I_inj=10.0, dt_ms=1.0, duration_ms=200.0
        )

        # Generate dummy current trace for stability report
        current_trace = np.ones_like(V_trace) * 10.0

        # Run stability diagnostic
        report = integration_stability_report(V_trace, current_trace, 1.0, "passive")

        assert report["is_stable"]
        assert report["voltage_finite"]["is_finite"]


class TestMembraneProperties:
    """Tests for membrane property calculations."""

    def test_tau_membrane_ms(self):
        """Test time constant calculation."""
        tau = tau_membrane_ms(C_m=100.0, g_L=1.0)
        assert np.isclose(tau, 100.0)

        tau = tau_membrane_ms(C_m=200.0, g_L=1.0)
        assert np.isclose(tau, 200.0)

    def test_tau_membrane_invalid(self):
        """Test error handling for invalid g_L."""
        with pytest.raises(ValueError):
            tau_membrane_ms(C_m=100.0, g_L=0.0)

        with pytest.raises(ValueError):
            tau_membrane_ms(C_m=100.0, g_L=-0.1)

    def test_steady_state_voltage_at_rest(self):
        """Test steady state when I_inj=0."""
        V_ss = steady_state_voltage(E_L=-65.0, g_L=1.0, I_inj=0.0)
        assert np.isclose(V_ss, -65.0)

    def test_steady_state_voltage_with_current(self):
        """Test steady state deflection with current."""
        V_ss = steady_state_voltage(E_L=-65.0, g_L=1.0, I_inj=10.0)
        expected = -65.0 + 10.0 / 1.0  # = -55.0 mV (depolarized)
        assert np.isclose(V_ss, expected)

    def test_steady_state_invalid(self):
        """Test error handling for invalid g_L."""
        with pytest.raises(ValueError):
            steady_state_voltage(E_L=-65.0, g_L=0.0, I_inj=10.0)

    def test_relaxation_curve(self):
        """Test exponential relaxation solution."""
        V_init = -65.0
        V_ss = -55.0
        tau = 100.0

        # At t=0, V should be V_init
        V_0 = relaxation_curve(0.0, V_init, V_ss, tau)
        assert np.isclose(V_0, V_init)

        # At t=tau, V should be 63.2% of way to V_ss
        V_tau = relaxation_curve(tau, V_init, V_ss, tau)
        expected_tau = V_ss + (V_init - V_ss) * np.exp(-1)
        assert np.isclose(V_tau, expected_tau)

        # At t=infinity, V should approach V_ss
        V_inf = relaxation_curve(1000 * tau, V_init, V_ss, tau)
        assert np.isclose(V_inf, V_ss, atol=1e-6)

    def test_relaxation_curve_array(self):
        """Test relaxation curve with array input."""
        t = np.array([0, 100, 200, 300])
        V_init = -65.0
        V_ss = -55.0
        tau = 100.0

        V_t = relaxation_curve(t, V_init, V_ss, tau)
        assert len(V_t) == 4
        assert np.isclose(V_t[0], V_init)

    def test_input_resistance(self):
        """Test input resistance calculation."""
        R_m = input_resistance_mohm(g_L=1.0)
        assert np.isclose(R_m, 1000.0)

        R_m = input_resistance_mohm(g_L=2.0)
        assert np.isclose(R_m, 500.0)

    def test_input_resistance_invalid(self):
        """Test error handling for invalid g_L."""
        with pytest.raises(ValueError):
            input_resistance_mohm(g_L=0.0)

    def test_membrane_potential_response(self):
        """Test step current response characterization."""
        resp = membrane_potential_response(I_step=10.0, g_L=1.0, tau=100.0)

        assert "V_ss" in resp
        assert "t_half" in resp
        assert "tau" in resp
        assert "is_finite" in resp

        # Check values
        assert np.isclose(resp["V_ss"], 10.0)  # 10 / 1.0
        assert np.isclose(resp["t_half"], 100.0 * np.log(2))
        assert resp["is_finite"]

    def test_membrane_potential_response_invalid(self):
        """Test error handling for invalid parameters."""
        with pytest.raises(ValueError):
            membrane_potential_response(I_step=10.0, g_L=0.0, tau=100.0)

        with pytest.raises(ValueError):
            membrane_potential_response(I_step=10.0, g_L=1.0, tau=0.0)


class TestPassiveMembranePhysics:
    """Tests for passive membrane physical behavior."""

    def test_rest_voltage_is_leak_potential(self):
        """Test that resting voltage equals leak potential."""
        params = PassiveMembraneParams(C_m=100.0, g_L=1.0, E_L=-70.0)
        V_trace = passive_membrane_simulate(
            V_init=-70.0, params=params, I_inj=0.0, dt_ms=1.0, duration_ms=500.0
        )
        assert np.allclose(V_trace, -70.0, atol=1e-6)

    def test_step_current_depolarizes(self):
        """Test that positive (inward) current depolarizes."""
        params = PassiveMembraneParams(C_m=100.0, g_L=1.0, E_L=-65.0)
        V_trace = passive_membrane_simulate(
            V_init=-65.0, params=params, I_inj=10.0, dt_ms=1.0, duration_ms=2000.0
        )
        V_final = V_trace[-1]
        # Check voltage reaches steady state
        V_ss = steady_state_voltage(params.E_L, params.g_L, 10.0)
        assert np.isclose(V_final, V_ss, atol=0.1)

    def test_passive_response_monotonic(self):
        """Test that response to constant current is monotonic (no oscillation)."""
        params = PassiveMembraneParams(C_m=100.0, g_L=1.0, E_L=-65.0)
        V_trace = passive_membrane_simulate(
            V_init=-65.0, params=params, I_inj=50.0, dt_ms=1.0, duration_ms=500.0
        )

        # Check that voltage is monotonically changing (no oscillation)
        dV = np.diff(V_trace)
        # All differences should have the same sign (no overshoot)
        assert np.all(dV > 0)  # All positive (depolarization for positive I_inj)


class TestPassiveMembraneNulls:
    """Tests for null/boundary conditions."""

    def test_timestep_stability_good(self):
        """Test that small dt produces stable, monotonic relaxation."""
        params = PassiveMembraneParams(C_m=100.0, g_L=1.0, E_L=-65.0)

        # Small timestep should be stable
        V_trace_stable = passive_membrane_simulate(
            V_init=-65.0,
            params=params,
            I_inj=10.0,
            dt_ms=10.0,  # Safe timestep
            duration_ms=500.0,
        )
        assert np.all(np.isfinite(V_trace_stable))
        # Should approach steady state monotonically
        dV = np.diff(V_trace_stable)
        assert np.all(dV > 0)  # All steps increase V

    def test_large_timestep_produces_large_jumps(self):
        """Test that very large dt produces unrealistic large jumps."""
        params = PassiveMembraneParams(C_m=100.0, g_L=1.0, E_L=-65.0)

        # Very large timestep will produce large jumps (may overshoot)
        V_trace = passive_membrane_simulate(
            V_init=-65.0,
            params=params,
            I_inj=10.0,
            dt_ms=500.0,  # Very large, unrealistic timestep
            duration_ms=1000.0,
        )

        # Trace should still be finite (forward Euler doesn't always explode)
        assert np.all(np.isfinite(V_trace))
        # But the amplitude of changes should be unrealistic
        max_change = np.max(np.diff(V_trace))
        assert max_change > 5.0  # Larger jumps than small dt
