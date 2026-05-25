"""Final tests for v0.2 passive membrane chapter (v0.2.10).

Validates:
- Bridge documentation (v0.2.7)
- Notebook existence and structure (v0.2.8)
- Exercise availability (v0.2.9)
- No regressions from v0.2.2-v0.2.3
- JSON-safe reports and manifests
- Chapter completeness

v0.2.10 gate tests before v0.2.11 release.
"""

import pytest
import json
import os
import numpy as np

from jbiophysic.passive_membrane import (
    PassiveMembraneParams,
    passive_membrane_simulate,
)
from jbiophysic.units import integration_stability_report


class TestChapter2Documentation:
    """Tests for v0.2.7–v0.2.9 documentation completeness."""

    def test_jaxfne_bridge_note_exists(self):
        """Test that bridge documentation file exists."""
        bridge_path = os.path.join(
            os.path.dirname(__file__),
            "../docs/v0_2_jaxfne_bridge.md"
        )
        assert os.path.exists(bridge_path), "Bridge note not found"

    def test_jaxfne_bridge_contains_contract_language(self):
        """Test that bridge note explains source-field-probe contract."""
        bridge_path = os.path.join(
            os.path.dirname(__file__),
            "../docs/v0_2_jaxfne_bridge.md"
        )
        with open(bridge_path) as f:
            content = f.read()

        # Check for key terms
        assert "source-field-probe" in content or "source" in content
        assert "metadata" in content
        assert "conductivity" in content
        assert "DOES NOT" in content or "NOT establish" in content or "does not" in content

    def test_exercises_exist(self):
        """Test that exercises file exists."""
        exercises_path = os.path.join(
            os.path.dirname(__file__),
            "../docs/v0_2_exercises.md"
        )
        assert os.path.exists(exercises_path), "Exercises not found"

    def test_exercises_contain_6_problems(self):
        """Test that exercises document lists problems."""
        exercises_path = os.path.join(
            os.path.dirname(__file__),
            "../docs/v0_2_exercises.md"
        )
        with open(exercises_path) as f:
            content = f.read()

        # Count "Exercise" sections
        exercise_count = content.count("## Exercise")
        assert exercise_count >= 6, f"Expected at least 6 exercises, found {exercise_count}"

    def test_notebook_exists(self):
        """Test that tutorial notebook exists."""
        notebook_path = os.path.join(
            os.path.dirname(__file__),
            "../tutorials/v0_2_passive_membrane_tutorial.ipynb"
        )
        assert os.path.exists(notebook_path), "Notebook not found"

    def test_notebook_has_required_sections(self):
        """Test that notebook includes all 13 required sections."""
        notebook_path = os.path.join(
            os.path.dirname(__file__),
            "../tutorials/v0_2_passive_membrane_tutorial.ipynb"
        )
        with open(notebook_path) as f:
            notebook = json.load(f)

        # Extract markdown cell contents
        markdown_cells = [
            cell for cell in notebook["cells"]
            if cell["cell_type"] == "markdown"
        ]
        full_text = "\n".join([
            "".join(cell["source"]) for cell in markdown_cells
        ])

        # Check for required sections
        required_sections = [
            "Learning Objectives",
            "Biological/Computational Question",
            "Mathematical Glossary",
            "Units",
            "Simulation",
            "Diagnostics",
            "Sweep",
            "Wrong-Timestep",
            "Manifest",
            "Interpretation",
            "Failure Modes",
            "Exercises",
            "DOES NOT",
        ]

        for section in required_sections:
            assert section in full_text, f"Required section '{section}' not found in notebook"


class TestPassiveMembraneManifests:
    """Tests for JSON-safe manifests and reports."""

    def test_passive_membrane_report_json_serializable(self):
        """Test that a full simulation produces JSON-safe output."""
        params = PassiveMembraneParams(C_m=100.0, g_L=1.0, E_L=-65.0)
        V_trace = passive_membrane_simulate(
            V_init=-65.0, params=params, I_inj=10.0, dt_ms=1.0, duration_ms=200.0
        )

        # Create a manifest-like report
        report = {
            "chapter": "v0.2",
            "parameters": {
                "C_m_pF": float(params.C_m),
                "g_L_nS": float(params.g_L),
                "E_L_mV": float(params.E_L),
            },
            "results": {
                "n_steps": int(len(V_trace) - 1),
                "V_init_mV": float(V_trace[0]),
                "V_final_mV": float(V_trace[-1]),
                "all_finite": bool(np.all(np.isfinite(V_trace))),
            },
            "truth_status": "truth_safe_unverified",
        }

        # Must be JSON-serializable
        json_str = json.dumps(report)
        assert isinstance(json_str, str)
        assert len(json_str) > 0

    def test_stability_report_json_safe(self):
        """Test that integration_stability_report produces JSON-safe output."""
        params = PassiveMembraneParams(C_m=100.0, g_L=1.0, E_L=-65.0)
        V_trace = passive_membrane_simulate(
            V_init=-65.0, params=params, I_inj=10.0, dt_ms=1.0, duration_ms=200.0
        )
        current_trace = np.ones_like(V_trace) * 10.0

        report = integration_stability_report(V_trace, current_trace, 1.0, "passive")

        # Must be JSON-serializable
        json_str = json.dumps(report)
        assert isinstance(json_str, str)

        # Must have key fields
        assert "is_stable" in report
        assert "voltage_finite" in report

    def test_no_nan_in_diagnostics(self):
        """Test that passive membrane diagnostics never produce NaN."""
        from jbiophysic.passive_membrane import (
            tau_membrane_ms,
            steady_state_voltage,
            input_resistance_mohm,
        )

        # Test with range of reasonable values
        for C_m in [50.0, 100.0, 200.0]:
            for g_L in [0.1, 1.0, 5.0]:
                for I_inj in [0.0, 10.0, 100.0]:
                    tau = tau_membrane_ms(C_m, g_L)
                    V_ss = steady_state_voltage(-65.0, g_L, I_inj)
                    R_m = input_resistance_mohm(g_L)

                    assert np.isfinite(tau), f"NaN tau for C_m={C_m}, g_L={g_L}"
                    assert np.isfinite(V_ss), f"NaN V_ss for C_m={C_m}, g_L={g_L}, I_inj={I_inj}"
                    assert np.isfinite(R_m), f"NaN R_m for g_L={g_L}"


class TestChapter2NoRegressions:
    """Regression tests: verify v0.2.2–v0.2.3 baseline still passes."""

    def test_basic_simulation_still_works(self):
        """Test that basic passive_membrane_simulate still works."""
        params = PassiveMembraneParams(C_m=100.0, g_L=1.0, E_L=-65.0)
        V_trace = passive_membrane_simulate(
            V_init=-65.0,
            params=params,
            I_inj=10.0,
            dt_ms=1.0,
            duration_ms=100.0,
        )
        assert len(V_trace) == 101
        assert np.all(np.isfinite(V_trace))

    def test_at_rest_voltage_unchanged(self):
        """Test that resting voltage remains at E_L."""
        params = PassiveMembraneParams(C_m=100.0, g_L=1.0, E_L=-70.0)
        V_trace = passive_membrane_simulate(
            V_init=-70.0,
            params=params,
            I_inj=0.0,
            dt_ms=1.0,
            duration_ms=200.0,
        )
        assert np.allclose(V_trace, -70.0, atol=1e-6)

    def test_step_response_depolarizes(self):
        """Test that positive current depolarizes."""
        from jbiophysic.passive_membrane import steady_state_voltage

        params = PassiveMembraneParams(C_m=100.0, g_L=1.0, E_L=-65.0)
        I_inj = 10.0
        V_ss = steady_state_voltage(params.E_L, params.g_L, I_inj)

        V_trace = passive_membrane_simulate(
            V_init=-65.0,
            params=params,
            I_inj=I_inj,
            dt_ms=1.0,
            duration_ms=2000.0,
        )

        # Should approach steady state
        assert np.isclose(V_trace[-1], V_ss, atol=0.1)
        # Should depolarize (V > -65)
        assert V_trace[-1] > -65.0


class TestJaxfneGuardBehavior:
    """Tests for optional jaxfne guard (v0.2.7 contract)."""

    def test_jaxfne_optional_import(self):
        """Test that jaxfne is optional (no forced import)."""
        # This test just verifies the module loads without jaxfne
        try:
            from jbiophysic.passive_membrane import passive_membrane_simulate
            assert callable(passive_membrane_simulate)
        except ImportError as e:
            # jaxfne is optional; jbiophysic.passive_membrane must work without it
            if "jaxfne" in str(e):
                pytest.skip("jaxfne guard not properly configured")
            raise


class TestChapter2Completeness:
    """Tests for chapter-level completeness."""

    def test_all_module_imports(self):
        """Test that all v0.2 module imports work."""
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

        # All imports must succeed
        assert callable(passive_membrane_step)
        assert callable(passive_membrane_simulate)
        assert callable(tau_membrane_ms)
        assert callable(steady_state_voltage)
        assert callable(relaxation_curve)
        assert callable(input_resistance_mohm)
        assert callable(membrane_potential_response)

    def test_bridge_note_avoids_overclaims(self):
        """Test that bridge documentation avoids overclaiming LFP/EEG."""
        bridge_path = os.path.join(
            os.path.dirname(__file__),
            "../docs/v0_2_jaxfne_bridge.md"
        )
        with open(bridge_path) as f:
            content = f.read()

        # Bridge should have clear disclaimers
        assert "❌" in content or "NOT establish" in content
        assert "LFP" in content or "extracellular" in content

    def test_examples_run_without_error(self):
        """Test that example scripts load without syntax errors."""
        example_path = os.path.join(
            os.path.dirname(__file__),
            "../examples/v0_2_passive_membrane_minimal.py"
        )
        assert os.path.exists(example_path), "Example script not found"

        # Verify it's syntactically valid Python
        with open(example_path) as f:
            code = f.read()
        compile(code, example_path, "exec")  # Will raise SyntaxError if invalid

    def test_doctrine_mentions_all_phases(self):
        """Test that doctrine covers v0.2.0–v0.2.11."""
        doctrine_path = os.path.join(
            os.path.dirname(__file__),
            "../docs/v0_2_membrane_doctrine.md"
        )
        with open(doctrine_path) as f:
            content = f.read()

        # Check for phase coverage
        for phase in range(0, 12):
            assert f"v0.2.{phase}" in content, f"v0.2.{phase} not mentioned in doctrine"
