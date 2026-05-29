"""Unit tests for advanced jaxfne integration module.

These tests require jaxfne to be installed:
    pip install -e '.[jaxfne]'
"""

import numpy as np
import pytest

# Skip entire test module if jaxfne not available
try:
    import jaxfne as jtfne  # noqa: F401
    JAXFNE_AVAILABLE = True
except ImportError:
    JAXFNE_AVAILABLE = False

pytestmark = pytest.mark.skipif(
    not JAXFNE_AVAILABLE,
    reason="jaxfne not installed (pip install -e '.[jaxfne]')"
)

if JAXFNE_AVAILABLE:
    from jbiophysic.jaxfne_advanced import (
        CustomReceptorSpec,
        analyze_critical_neurons,
        build_multi_area_edges,
        compute_connection_motifs,
        optimize_synaptic_weights,
    )
    from jbiophysic.jtfne import JTFNEInitConfig, construct


@pytest.fixture
def test_model():
    """Create a test model with jaxfne objects."""
    init = JTFNEInitConfig(n_neuron_per_column=30, seed=42, area_order=("V1", "V4"))
    return construct(init, include_jaxfne=True)


class TestCustomReceptorSpec:
    """Test custom receptor kinetics specification."""

    def test_receptor_spec_creation(self):
        """Test that CustomReceptorSpec initializes correctly."""
        spec = CustomReceptorSpec()
        assert spec is not None

    def test_receptor_spec_get_tau(self):
        """Test retrieving tau values."""
        spec = CustomReceptorSpec()

        # Get standard AMPA tau
        tau_ampa = spec.get_tau("E", "E", receptor_index=0)
        assert tau_ampa == 2.0

        # Get standard GABA_A tau
        tau_gaba = spec.get_tau("PV", "E", receptor_index=1)
        assert tau_gaba == 5.0

    def test_receptor_spec_override(self):
        """Test overriding tau for specific connections."""
        spec = CustomReceptorSpec()

        # Override E→PV AMPA tau
        spec.set_tau("E", "PV", tau_ms=3.0)

        tau = spec.get_tau("E", "PV", receptor_index=0)
        assert tau == 3.0

        # Non-overridden should still be standard
        tau_default = spec.get_tau("E", "E", receptor_index=0)
        assert tau_default == 2.0


class TestMultiAreaEdges:
    """Test multi-area routing with custom receptor kinetics."""

    def test_build_multi_area_edges_basic(self, test_model):
        """Test basic multi-area edge building."""
        edges = build_multi_area_edges(
            test_model.eig_network,
            test_model.neurons,
            area_order=("V1", "V4"),
            existing_edges=test_model.edges,
        )

        assert edges.pre.shape[0] > 0
        assert len(edges.pre) == len(edges.post)
        assert len(edges.weight) == len(edges.pre)
        assert len(edges.tau_ms) == len(edges.pre)

    def test_build_with_custom_kinetics(self, test_model):
        """Test multi-area edges with custom receptor specs."""
        spec = CustomReceptorSpec()
        spec.set_tau("E", "PV", tau_ms=3.0)

        edges = build_multi_area_edges(
            test_model.eig_network,
            test_model.neurons,
            area_order=("V1", "V4"),
            receptor_kinetics=spec,
            existing_edges=test_model.edges,
        )

        assert edges.tau_ms is not None
        assert np.any(edges.tau_ms > 0)


class TestSynapticWeightOptimization:
    """Test synaptic weight scaling."""

    def test_weight_scaling(self, test_model):
        """Test that weights are properly scaled."""
        original_weights = np.asarray(test_model.edges.weight)
        mean_original = np.mean(np.abs(original_weights))

        # Scale up
        scaled_edges = optimize_synaptic_weights(test_model.edges, scale_factor=2.0)
        scaled_weights = np.asarray(scaled_edges.weight)
        mean_scaled = np.mean(np.abs(scaled_weights))

        # Check scaling
        assert mean_scaled > mean_original

    def test_weight_scaling_preserves_structure(self, test_model):
        """Test that scaling preserves edge structure."""
        original_edges = test_model.edges

        scaled_edges = optimize_synaptic_weights(original_edges, scale_factor=1.5)

        # Structure should be identical
        np.testing.assert_array_equal(scaled_edges.pre, original_edges.pre)
        np.testing.assert_array_equal(scaled_edges.post, original_edges.post)
        np.testing.assert_array_equal(
            scaled_edges.receptor_index, original_edges.receptor_index
        )


class TestConnectionMotifs:
    """Test network motif detection."""

    def test_motif_computation(self, test_model):
        """Test that motifs are computed correctly."""
        motifs = compute_connection_motifs(test_model.eig_network, test_model.edges)

        assert "n_edges" in motifs
        assert "reciprocal_pairs" in motifs
        assert "triangles" in motifs
        assert "e_to_e" in motifs
        assert "e_to_i" in motifs
        assert "i_to_e" in motifs
        assert "i_to_i" in motifs

        # Basic sanity checks
        assert motifs["n_edges"] > 0
        assert motifs["reciprocal_pairs"] >= 0
        assert motifs["e_to_e"] + motifs["e_to_i"] + motifs["i_to_e"] + motifs["i_to_i"] == motifs["n_edges"]

    def test_motif_exc_inh_breakdown(self, test_model):
        """Test that excitatory/inhibitory breakdown is consistent."""
        motifs = compute_connection_motifs(test_model.eig_network, test_model.edges)

        # Sum of all connection types should equal total edges
        total = motifs["e_to_e"] + motifs["e_to_i"] + motifs["i_to_e"] + motifs["i_to_i"]
        assert total == motifs["n_edges"]


class TestCriticalNeurons:
    """Test hub neuron analysis."""

    def test_critical_neuron_analysis(self, test_model):
        """Test hub neuron identification."""
        critical = analyze_critical_neurons(test_model.eig_network, test_model.edges)

        assert "n_hubs" in critical
        assert "hub_fraction" in critical
        assert "hub_types" in critical
        assert "max_in_degree" in critical
        assert "max_out_degree" in critical
        assert "avg_in_degree" in critical
        assert "avg_out_degree" in critical

        # Basic sanity checks
        assert 0 <= critical["hub_fraction"] <= 1.0
        assert critical["max_in_degree"] >= critical["avg_in_degree"]
        assert critical["max_out_degree"] >= critical["avg_out_degree"]

    def test_hub_distribution(self, test_model):
        """Test that hub types are reasonable."""
        critical = analyze_critical_neurons(test_model.eig_network, test_model.edges)

        hub_types = critical["hub_types"]
        total_hubs = sum(hub_types.values())

        assert total_hubs == critical["n_hubs"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
