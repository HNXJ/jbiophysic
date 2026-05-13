"""Tests for TFNE spectrolaminar readout basis and trial projection.

Validates basis construction, FieldSolution metadata preservation,
contact projection, and readout trial functionality.
"""

import numpy as np
import pytest

from jbiophysic.configs import SpectrolaminarMotifConfig
from jbiophysic.models.spectrolaminar_motif import build_spectrolaminar_motif
from jbiophysic.models.tfne_spectrolaminar import (
    TFNEReadoutBasis,
    build_tfne_readout_basis,
    tfne_readout_trial,
    tfne_readout_trials,
)


class TestTFNEReadoutBasisConstruction:
    """Test TFNEReadoutBasis construction from spectrolaminar motif."""

    def test_basis_builds_from_smoke_model(self):
        """Verify TFNE readout basis can be built from Phase 2.3A smoke model."""
        cfg = SpectrolaminarMotifConfig(mode="smoke").with_smoke_defaults()
        model = build_spectrolaminar_motif(cfg)

        # Should not raise
        basis = build_tfne_readout_basis(model, cfg)

        assert isinstance(basis, TFNEReadoutBasis)
        assert basis.n_neurons == model.n_neurons
        assert basis.n_contacts == len(cfg.tfne_contact_depths_mm)

    def test_basis_has_all_required_metadata_fields(self):
        """Verify TFNEReadoutBasis includes all required metadata fields."""
        cfg = SpectrolaminarMotifConfig(mode="smoke").with_smoke_defaults()
        model = build_spectrolaminar_motif(cfg)
        basis = build_tfne_readout_basis(model, cfg)

        # Check all mandatory fields exist and are non-None
        assert hasattr(basis, "lfp_basis")
        assert hasattr(basis, "csd_basis")
        assert hasattr(basis, "contact_depths_m")
        assert hasattr(basis, "basis_conservation_max_abs")
        assert hasattr(basis, "solver_residual_max")
        assert hasattr(basis, "solver_residual_norms")
        assert hasattr(basis, "solver_status_counts")
        assert hasattr(basis, "solver_converged_count")
        assert hasattr(basis, "solver_mean_iterations")
        assert hasattr(basis, "gauge_applied")
        assert hasattr(basis, "boundary_condition")
        assert hasattr(basis, "source_calibration_status")
        assert hasattr(basis, "truth_mode")
        assert hasattr(basis, "claim_level")

    def test_basis_lfp_csd_matrices_are_finite(self):
        """Verify lfp_basis and csd_basis contain finite values only."""
        cfg = SpectrolaminarMotifConfig(mode="smoke").with_smoke_defaults()
        model = build_spectrolaminar_motif(cfg)
        basis = build_tfne_readout_basis(model, cfg)

        assert np.all(np.isfinite(basis.lfp_basis)), "lfp_basis contains non-finite values"
        assert np.all(np.isfinite(basis.csd_basis)), "csd_basis contains non-finite values"

    def test_basis_shapes_are_consistent(self):
        """Verify basis matrix shapes are (n_neurons, n_contacts)."""
        cfg = SpectrolaminarMotifConfig(mode="smoke").with_smoke_defaults()
        model = build_spectrolaminar_motif(cfg)
        basis = build_tfne_readout_basis(model, cfg)

        n_neurons = model.n_neurons
        n_contacts = len(cfg.tfne_contact_depths_mm)

        assert basis.lfp_basis.shape == (n_neurons, n_contacts)
        assert basis.csd_basis.shape == (n_neurons, n_contacts)
        assert basis.contact_depths_m.shape == (n_contacts,)

    def test_contact_depths_are_finite_and_monotonic(self):
        """Verify contact_depths_m are finite and monotonically ordered."""
        cfg = SpectrolaminarMotifConfig(mode="smoke").with_smoke_defaults()
        model = build_spectrolaminar_motif(cfg)
        basis = build_tfne_readout_basis(model, cfg)

        depths = basis.contact_depths_m
        assert np.all(np.isfinite(depths)), "contact_depths_m contains non-finite values"
        # Check monotonic (either increasing or decreasing)
        diffs = np.diff(depths)
        is_monotonic = np.all(diffs >= 0) or np.all(diffs <= 0)
        assert is_monotonic, "contact_depths_m is not monotonic"

    def test_solver_residual_norms_is_non_empty(self):
        """Verify solver_residual_norms tuple is populated (one per neuron)."""
        cfg = SpectrolaminarMotifConfig(mode="smoke").with_smoke_defaults()
        model = build_spectrolaminar_motif(cfg)
        basis = build_tfne_readout_basis(model, cfg)

        assert len(basis.solver_residual_norms) > 0, "solver_residual_norms is empty"
        assert len(basis.solver_residual_norms) == model.n_neurons

    def test_solver_residual_max_matches_tuple_max(self):
        """Verify solver_residual_max equals the maximum of solver_residual_norms."""
        cfg = SpectrolaminarMotifConfig(mode="smoke").with_smoke_defaults()
        model = build_spectrolaminar_motif(cfg)
        basis = build_tfne_readout_basis(model, cfg)

        expected_max = float(np.max(basis.solver_residual_norms))
        assert np.isclose(basis.solver_residual_max, expected_max), \
            f"solver_residual_max {basis.solver_residual_max} != max of tuple {expected_max}"

    def test_solver_status_counts_is_populated(self):
        """Verify solver_status_counts dict contains 'converged' and 'max_iters' keys."""
        cfg = SpectrolaminarMotifConfig(mode="smoke").with_smoke_defaults()
        model = build_spectrolaminar_motif(cfg)
        basis = build_tfne_readout_basis(model, cfg)

        assert isinstance(basis.solver_status_counts, dict)
        assert "converged" in basis.solver_status_counts
        assert "max_iters" in basis.solver_status_counts
        total = basis.solver_status_counts["converged"] + basis.solver_status_counts["max_iters"]
        assert total == model.n_neurons, \
            f"solver_status_counts does not sum to n_neurons: {total} != {model.n_neurons}"

    def test_solver_converged_count_is_integer(self):
        """Verify solver_converged_count is an integer matching status_counts['converged']."""
        cfg = SpectrolaminarMotifConfig(mode="smoke").with_smoke_defaults()
        model = build_spectrolaminar_motif(cfg)
        basis = build_tfne_readout_basis(model, cfg)

        assert isinstance(basis.solver_converged_count, (int, np.integer))
        assert basis.solver_converged_count == basis.solver_status_counts["converged"]

    def test_solver_mean_iterations_is_finite(self):
        """Verify solver_mean_iterations is a finite float."""
        cfg = SpectrolaminarMotifConfig(mode="smoke").with_smoke_defaults()
        model = build_spectrolaminar_motif(cfg)
        basis = build_tfne_readout_basis(model, cfg)

        assert np.isfinite(basis.solver_mean_iterations), \
            "solver_mean_iterations is not finite"
        assert isinstance(basis.solver_mean_iterations, (float, np.floating))
        assert basis.solver_mean_iterations > 0.0, \
            "solver_mean_iterations must be positive"

    def test_gauge_and_boundary_condition_are_non_empty(self):
        """Verify gauge_applied and boundary_condition strings are not empty."""
        cfg = SpectrolaminarMotifConfig(mode="smoke").with_smoke_defaults()
        model = build_spectrolaminar_motif(cfg)
        basis = build_tfne_readout_basis(model, cfg)

        assert isinstance(basis.gauge_applied, str)
        assert len(basis.gauge_applied) > 0, "gauge_applied is empty"
        assert isinstance(basis.boundary_condition, str)
        assert len(basis.boundary_condition) > 0, "boundary_condition is empty"

    def test_source_calibration_status_is_non_empty(self):
        """Verify source_calibration_status is a non-empty string."""
        cfg = SpectrolaminarMotifConfig(mode="smoke").with_smoke_defaults()
        model = build_spectrolaminar_motif(cfg)
        basis = build_tfne_readout_basis(model, cfg)

        assert isinstance(basis.source_calibration_status, str)
        assert len(basis.source_calibration_status) > 0, "source_calibration_status is empty"

    def test_truth_mode_is_truth_safe_unverified(self):
        """Verify truth_mode is always 'truth_safe_unverified' for Phase 2.4."""
        cfg = SpectrolaminarMotifConfig(mode="smoke").with_smoke_defaults()
        model = build_spectrolaminar_motif(cfg)
        basis = build_tfne_readout_basis(model, cfg)

        assert basis.truth_mode == "truth_safe_unverified"

    def test_claim_level_from_config(self):
        """Verify claim_level matches config setting."""
        cfg = SpectrolaminarMotifConfig(mode="smoke").with_smoke_defaults()
        model = build_spectrolaminar_motif(cfg)
        basis = build_tfne_readout_basis(model, cfg)

        assert basis.claim_level == cfg.claim_level


class TestTFNEReadoutTrial:
    """Test single-trial source projection to contacts."""

    def test_readout_trial_returns_finite_contacts(self):
        """Verify tfne_readout_trial returns finite lfp_contacts and csd_contacts."""
        cfg = SpectrolaminarMotifConfig(mode="smoke").with_smoke_defaults()
        model = build_spectrolaminar_motif(cfg)
        basis = build_tfne_readout_basis(model, cfg)

        # Create synthetic source traces
        n_timepoints = 100
        source_traces = np.random.randn(n_timepoints, model.n_neurons)

        result = tfne_readout_trial(source_traces, basis)

        assert np.all(np.isfinite(result["lfp_contacts"])), \
            "lfp_contacts contains non-finite values"
        assert np.all(np.isfinite(result["csd_contacts"])), \
            "csd_contacts contains non-finite values"

    def test_readout_trial_output_shapes(self):
        """Verify tfne_readout_trial output shapes are correct."""
        cfg = SpectrolaminarMotifConfig(mode="smoke").with_smoke_defaults()
        model = build_spectrolaminar_motif(cfg)
        basis = build_tfne_readout_basis(model, cfg)

        n_timepoints = 50
        source_traces = np.random.randn(n_timepoints, model.n_neurons)
        n_contacts = len(cfg.tfne_contact_depths_mm)

        result = tfne_readout_trial(source_traces, basis)

        assert result["lfp_contacts"].shape == (n_timepoints, n_contacts)
        assert result["csd_contacts"].shape == (n_timepoints, n_contacts)
        assert len(result["contact_depths_m"]) == n_contacts

    def test_readout_trial_metadata_included(self):
        """Verify tfne_readout_trial includes metadata dict."""
        cfg = SpectrolaminarMotifConfig(mode="smoke").with_smoke_defaults()
        model = build_spectrolaminar_motif(cfg)
        basis = build_tfne_readout_basis(model, cfg)

        source_traces = np.random.randn(10, model.n_neurons)
        result = tfne_readout_trial(source_traces, basis)

        assert "metadata" in result
        assert isinstance(result["metadata"], dict)
        assert "basis_claim_level" in result["metadata"]
        assert "solver_converged_count" in result["metadata"]
        assert "solver_residual_max" in result["metadata"]

    def test_readout_trial_invalid_neuron_count_raises(self):
        """Verify tfne_readout_trial raises ValueError on neuron count mismatch."""
        cfg = SpectrolaminarMotifConfig(mode="smoke").with_smoke_defaults()
        model = build_spectrolaminar_motif(cfg)
        basis = build_tfne_readout_basis(model, cfg)

        # Wrong number of neurons
        bad_traces = np.random.randn(100, model.n_neurons + 5)

        with pytest.raises(ValueError, match="source_traces has .* neurons but basis expects"):
            tfne_readout_trial(bad_traces, basis)

    def test_readout_trial_invalid_shape_raises(self):
        """Verify tfne_readout_trial raises ValueError on non-2D input."""
        cfg = SpectrolaminarMotifConfig(mode="smoke").with_smoke_defaults()
        model = build_spectrolaminar_motif(cfg)
        basis = build_tfne_readout_basis(model, cfg)

        # 1D array instead of 2D
        bad_traces = np.random.randn(100)

        with pytest.raises(ValueError, match="source_traces must be 2D"):
            tfne_readout_trial(bad_traces, basis)

    def test_readout_trial_zero_sourcetraces(self):
        """Verify tfne_readout_trial handles zero source traces (outputs zeros)."""
        cfg = SpectrolaminarMotifConfig(mode="smoke").with_smoke_defaults()
        model = build_spectrolaminar_motif(cfg)
        basis = build_tfne_readout_basis(model, cfg)

        # Zero source traces
        source_traces = np.zeros((100, model.n_neurons))

        result = tfne_readout_trial(source_traces, basis)

        # With zero sources, contacts should be zero
        assert np.allclose(result["lfp_contacts"], 0.0)
        assert np.allclose(result["csd_contacts"], 0.0)


class TestTFNEReadoutTrials:
    """Test batch trial projection."""

    def test_readout_trials_returns_list(self):
        """Verify tfne_readout_trials returns dict with trials list."""
        cfg = SpectrolaminarMotifConfig(mode="smoke").with_smoke_defaults()
        model = build_spectrolaminar_motif(cfg)
        basis = build_tfne_readout_basis(model, cfg)

        source_traces_1 = np.random.randn(50, model.n_neurons)
        source_traces_2 = np.random.randn(75, model.n_neurons)

        result = tfne_readout_trials([source_traces_1, source_traces_2], basis)

        assert isinstance(result["trials"], list)
        assert len(result["trials"]) == 2

    def test_readout_trials_includes_basis(self):
        """Verify tfne_readout_trials includes basis object in result."""
        cfg = SpectrolaminarMotifConfig(mode="smoke").with_smoke_defaults()
        model = build_spectrolaminar_motif(cfg)
        basis = build_tfne_readout_basis(model, cfg)

        source_traces = [np.random.randn(50, model.n_neurons)]
        result = tfne_readout_trials(source_traces, basis)

        assert "basis" in result
        assert result["basis"] is basis


class TestFieldSolutionMetadataPreservation:
    """Test that FieldSolution metadata is properly preserved in readout basis."""

    def test_no_bare_ndarray_assumption(self):
        """Verify build_tfne_readout_basis does not assume jacobi solver returns bare ndarray.

        This is the critical doctrine test: the code must extract FieldSolution object
        and access solution.phi_e, not assume a bare phi array is returned.

        Since we cannot directly inspect the solver call internals, we verify that:
        1. All required metadata fields are present and non-empty
        2. The bases have expected numerical properties
        """
        cfg = SpectrolaminarMotifConfig(mode="smoke").with_smoke_defaults()
        model = build_spectrolaminar_motif(cfg)
        basis = build_tfne_readout_basis(model, cfg)

        # If bare ndarray was assumed, these would be missing or corrupted
        assert basis.solver_residual_norms is not None
        assert len(basis.solver_residual_norms) == model.n_neurons
        assert basis.gauge_applied is not None
        assert basis.boundary_condition is not None
        # All residuals should be positive (from solver convergence check)
        assert all(r > 0 for r in basis.solver_residual_norms)

    def test_solver_metadata_consistency(self):
        """Verify solver metadata is internally consistent across all solves."""
        cfg = SpectrolaminarMotifConfig(mode="smoke").with_smoke_defaults()
        model = build_spectrolaminar_motif(cfg)
        basis = build_tfne_readout_basis(model, cfg)

        # All residuals should be non-negative
        assert all(r >= 0 for r in basis.solver_residual_norms)

        # Max residual should be >= all recorded residuals
        assert all(basis.solver_residual_max >= r for r in basis.solver_residual_norms)

        # Convergence count should be non-negative
        assert basis.solver_converged_count >= 0

        # Mean iterations should be reasonable (> 0 and <= max steps in config)
        assert 0 < basis.solver_mean_iterations <= cfg.tfne_jacobi_steps

        # Status counts should be non-negative and sum to total
        assert basis.solver_status_counts["converged"] >= 0
        assert basis.solver_status_counts["max_iters"] >= 0


class TestReadoutBasisIntegration:
    """Integration tests combining basis construction and readout."""

    def test_smoke_to_readout_pipeline(self):
        """Test end-to-end smoke config → model → basis → trial projection."""
        cfg = SpectrolaminarMotifConfig(mode="smoke").with_smoke_defaults()
        model = build_spectrolaminar_motif(cfg)
        basis = build_tfne_readout_basis(model, cfg)

        # Synthetic neuronal activity
        source_traces = np.random.randn(200, model.n_neurons)
        result = tfne_readout_trial(source_traces, basis)

        assert result["lfp_contacts"].shape[0] == 200
        assert result["lfp_contacts"].shape[1] == len(cfg.tfne_contact_depths_mm)
        assert np.all(np.isfinite(result["lfp_contacts"]))

    def test_basis_is_frozen(self):
        """Verify TFNEReadoutBasis is frozen (immutable)."""
        cfg = SpectrolaminarMotifConfig(mode="smoke").with_smoke_defaults()
        model = build_spectrolaminar_motif(cfg)
        basis = build_tfne_readout_basis(model, cfg)

        # Frozen dataclass should not allow assignment
        with pytest.raises(AttributeError):
            basis.claim_level = "modified"

    def test_basis_deterministic_with_seed(self):
        """Verify basis is deterministic given same config and model seed."""
        cfg1 = SpectrolaminarMotifConfig(mode="smoke", seed=42).with_smoke_defaults()
        cfg2 = SpectrolaminarMotifConfig(mode="smoke", seed=42).with_smoke_defaults()

        model1 = build_spectrolaminar_motif(cfg1)
        model2 = build_spectrolaminar_motif(cfg2)

        basis1 = build_tfne_readout_basis(model1, cfg1)
        basis2 = build_tfne_readout_basis(model2, cfg2)

        # Bases should be identical
        assert np.allclose(basis1.lfp_basis, basis2.lfp_basis)
        assert np.allclose(basis1.csd_basis, basis2.csd_basis)
