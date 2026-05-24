"""Unit tests for jaxfne integration layer."""

import numpy as np
import pytest

from jbiophysic.jaxfne_integration import (
    jbiophysic_to_eig_network,
    project_to_laminar_field,
    simulate_with_jaxfne,
)
from jbiophysic.jtfne import JTFNEInitConfig, construct


@pytest.fixture
def minimal_model():
    """Create a minimal jbiophysic model for testing."""
    init = JTFNEInitConfig(
        n_neuron_per_column=30,
        seed=42,
        area_order=("V1", "V4"),
    )
    return construct(init)


@pytest.fixture
def eig_network_with_edges(minimal_model):
    """Create EIGNetwork and EdgeList from minimal model."""
    eig_net, edges = jbiophysic_to_eig_network(minimal_model, use_receptor_exponential=True)
    return eig_net, edges


class TestConversionBasics:
    """Test basic conversion from jbiophysic to jaxfne."""

    def test_eig_network_creation(self, minimal_model):
        """Test that EIGNetwork is created with correct shapes."""
        eig_net, edges = jbiophysic_to_eig_network(minimal_model, use_receptor_exponential=True)

        n_neurons = len(minimal_model.neurons)
        assert eig_net.params.a.shape == (n_neurons,)
        assert eig_net.params.b.shape == (n_neurons,)
        assert eig_net.params.c.shape == (n_neurons,)
        assert eig_net.params.d.shape == (n_neurons,)
        assert eig_net.params.drive.shape == (n_neurons,)
        assert eig_net.params.v0.shape == (n_neurons,)
        assert eig_net.params.u0.shape == (n_neurons,)
        assert eig_net.params.sign.shape == (n_neurons,)
        assert eig_net.params.source_scale.shape == (n_neurons,)
        assert eig_net.positions.shape == (n_neurons, 3)
        assert len(eig_net.params.labels) == n_neurons
        assert len(eig_net.params.layer_labels) == n_neurons

    def test_parameter_value_preservation(self, minimal_model):
        """Test that Izhikevich parameters are correctly extracted."""
        eig_net, _ = jbiophysic_to_eig_network(minimal_model, use_receptor_exponential=True)

        # Check that parameters match expected ranges
        assert np.all(eig_net.params.a > 0)
        assert np.all(eig_net.params.b > 0)
        assert np.all(eig_net.params.c < 0)  # Reset potential
        assert np.all(eig_net.params.d > 0)

    def test_position_normalization(self, minimal_model):
        """Test that positions are normalized to [0, 1] in depth."""
        eig_net, _ = jbiophysic_to_eig_network(minimal_model, use_receptor_exponential=True)

        # Third coordinate should be laminar depth [0, 1]
        depth = eig_net.positions[:, 2]
        assert np.all(depth >= 0.0)
        assert np.all(depth <= 1.0)

    def test_cell_type_labels(self, minimal_model):
        """Test that cell type labels are preserved."""
        eig_net, _ = jbiophysic_to_eig_network(minimal_model, use_receptor_exponential=True)

        labels = eig_net.params.labels
        unique_types = set(labels)
        # Should have expected cell types
        assert len(unique_types) > 0
        for ct in unique_types:
            assert ct in ("E", "PV", "SST", "VIP")

    def test_edge_list_creation(self, eig_network_with_edges):
        """Test that EdgeList is created with correct structure."""
        _, edges = eig_network_with_edges

        n_edges = len(edges.pre)
        assert n_edges > 0
        assert edges.pre.shape == (n_edges,)
        assert edges.post.shape == (n_edges,)
        assert edges.weight.shape == (n_edges,)
        assert edges.receptor_index.shape == (n_edges,)
        assert edges.tau_ms.shape == (n_edges,)

    def test_receptor_indices_valid(self, eig_network_with_edges):
        """Test that receptor indices are valid (0-3)."""
        _, edges = eig_network_with_edges

        unique_receptors = np.unique(edges.receptor_index)
        assert np.all(unique_receptors >= 0)
        assert np.all(unique_receptors <= 3)


class TestSimulation:
    """Test simulation using jaxfne backend."""

    def test_simulation_output_shapes(self, eig_network_with_edges):
        """Test that simulation returns correct output shapes."""
        eig_net, edges = eig_network_with_edges
        n_neurons = len(eig_net.params.a)
        n_steps = 50

        v, u, spikes = simulate_with_jaxfne(
            eig_net,
            edges,
            n_steps=n_steps,
            dt_ms=0.5,
            seed=42,
            use_receptor_exponential=True,
        )

        assert v.shape == (n_steps, n_neurons)
        assert u.shape == (n_steps, n_neurons)
        assert spikes.shape == (n_steps, n_neurons)

    def test_simulation_determinism(self, eig_network_with_edges):
        """Test that same seed produces identical spike raster."""
        eig_net, edges = eig_network_with_edges

        v1, u1, spikes1 = simulate_with_jaxfne(
            eig_net,
            edges,
            n_steps=100,
            dt_ms=0.5,
            seed=42,
            use_receptor_exponential=True,
        )

        v2, u2, spikes2 = simulate_with_jaxfne(
            eig_net,
            edges,
            n_steps=100,
            dt_ms=0.5,
            seed=42,
            use_receptor_exponential=True,
        )

        np.testing.assert_allclose(v1, v2, rtol=1e-5)
        np.testing.assert_allclose(u1, u2, rtol=1e-5)
        np.testing.assert_allclose(spikes1, spikes2, rtol=1e-5)

    def test_simulation_different_seeds(self, eig_network_with_edges):
        """Test that different seeds produce different results."""
        eig_net, edges = eig_network_with_edges

        spikes1 = simulate_with_jaxfne(
            eig_net,
            edges,
            n_steps=100,
            dt_ms=0.5,
            seed=42,
            use_receptor_exponential=True,
        )[2]

        spikes2 = simulate_with_jaxfne(
            eig_net,
            edges,
            n_steps=100,
            dt_ms=0.5,
            seed=43,
            use_receptor_exponential=True,
        )[2]

        # Should not be identical
        assert not np.allclose(spikes1, spikes2)

    def test_voltage_bounds(self, eig_network_with_edges):
        """Test that voltage stays within physiologically reasonable bounds."""
        eig_net, edges = eig_network_with_edges

        v, _, _ = simulate_with_jaxfne(
            eig_net,
            edges,
            n_steps=100,
            dt_ms=0.5,
            seed=42,
            use_receptor_exponential=True,
        )

        # Voltage should stay within some reasonable range for Izhikevich
        assert np.all(v >= -100.0)  # Well below typical reset
        assert np.all(v <= 50.0)  # Well above typical spike threshold


class TestFieldProjection:
    """Test laminar field projection using jaxfne."""

    def test_field_output_shapes(self, eig_network_with_edges):
        """Test that field projection returns correct output shapes."""
        eig_net, edges = eig_network_with_edges

        _, _, spikes = simulate_with_jaxfne(
            eig_net,
            edges,
            n_steps=50,
            dt_ms=0.5,
            seed=42,
            use_receptor_exponential=True,
        )

        source = spikes.astype(np.float32)
        field = project_to_laminar_field(
            source,
            eig_net.positions,
            n_contacts=16,
            width=0.1,
        )

        n_steps = 50
        n_contacts = 16
        assert field.source_proxy.shape == (n_steps, n_contacts)
        assert field.csd_proxy.shape == (n_steps, n_contacts)
        assert field.lfp_proxy.shape == (n_steps, n_contacts)
        assert field.contact_depths.shape == (n_contacts,)

    def test_field_contact_depths_valid(self, eig_network_with_edges):
        """Test that contact depths are properly normalized."""
        eig_net, edges = eig_network_with_edges

        _, _, spikes = simulate_with_jaxfne(
            eig_net,
            edges,
            n_steps=50,
            dt_ms=0.5,
            seed=42,
            use_receptor_exponential=True,
        )

        source = spikes.astype(np.float32)
        field = project_to_laminar_field(
            source,
            eig_net.positions,
            n_contacts=16,
            width=0.1,
        )

        # Contact depths should be within [0, 1]
        assert np.all(field.contact_depths >= 0.0)
        assert np.all(field.contact_depths <= 1.0)
        # Should be monotonically increasing
        assert np.all(np.diff(field.contact_depths) > 0)


class TestBackwardCompatibility:
    """Test that conversion preserves key properties."""

    def test_neuron_count_preserved(self, minimal_model):
        """Test that neuron count is preserved during conversion."""
        expected_n = len(minimal_model.neurons)
        eig_net, _ = jbiophysic_to_eig_network(minimal_model, use_receptor_exponential=True)

        actual_n = len(eig_net.params.a)
        assert actual_n == expected_n

    def test_connectivity_exists(self, eig_network_with_edges):
        """Test that some connectivity is preserved."""
        eig_net, edges = eig_network_with_edges

        n_neurons = len(eig_net.params.a)
        max_edges = n_neurons * (n_neurons - 1)  # Theoretical max

        actual_edges = len(edges.pre)
        assert actual_edges > 0
        assert actual_edges <= max_edges

    def test_nonzero_weights(self, eig_network_with_edges):
        """Test that synapse weights are nonzero."""
        _, edges = eig_network_with_edges

        assert np.all(np.abs(edges.weight) > 0)


class TestSimulateWithJaxfneBackend:
    """Test integration of jaxfne backend into jtfne.simulate()."""

    def test_simulate_jaxfne_backend(self):
        """Test that simulate() can use jaxfne backend."""
        from jbiophysic.jtfne import JTFNEInitConfig, construct, default_cfg, simulate
        from dataclasses import replace

        # Create minimal model
        init = JTFNEInitConfig(n_neuron_per_column=20, seed=42, area_order=("V1", "V4"))
        model = construct(init)

        cfg = default_cfg()
        sim = replace(cfg.sim, n_trials=1, t_ms=50.0, dt_ms=0.5)

        # Test jaxfne backend
        result = simulate(model, sim, backend="jaxfne")

        assert hasattr(result, "backend")
        assert result.backend == "jaxfne"
        assert len(result.trials) == 1

    def test_simulate_backend_output_shapes(self):
        """Test that jaxfne backend produces correct output shapes."""
        from jbiophysic.jtfne import JTFNEInitConfig, construct, default_cfg, simulate
        from dataclasses import replace

        init = JTFNEInitConfig(n_neuron_per_column=20, seed=42, area_order=("V1", "V4"))
        model = construct(init)

        cfg = default_cfg()
        sim = replace(cfg.sim, n_trials=1, t_ms=50.0, dt_ms=0.5, n_contacts=16)

        result = simulate(model, sim, backend="jaxfne")
        trial = result.trials[0]

        # Check structure
        assert "V1" in trial
        assert "V4" in trial

        for area in ["V1", "V4"]:
            area_data = trial[area]
            assert "spikes" in area_data
            assert "voltage_mV" in area_data
            assert "lfp_contacts" in area_data
            assert "csd_contacts" in area_data
            assert "contact_depths_m" in area_data
            assert "neurons" in area_data

            # Check shapes
            n_steps = 100  # 50ms / 0.5ms
            n_neurons = len(area_data["neurons"])
            assert area_data["spikes"].shape == (n_steps, n_neurons)
            assert area_data["voltage_mV"].shape == (n_steps, n_neurons)
            assert area_data["lfp_contacts"].shape == (n_steps, 16)
            assert area_data["csd_contacts"].shape == (n_steps, 16)

    def test_simulate_backend_legacy_vs_jaxfne(self):
        """Test that both backends produce reasonable outputs."""
        from jbiophysic.jtfne import JTFNEInitConfig, construct, default_cfg, simulate
        from dataclasses import replace

        init = JTFNEInitConfig(n_neuron_per_column=20, seed=42, area_order=("V1",))
        model = construct(init)

        cfg = default_cfg()
        sim = replace(cfg.sim, n_trials=1, t_ms=100.0, dt_ms=0.5)

        # Run both backends
        legacy = simulate(model, sim, backend="legacy")
        jaxfne_res = simulate(model, sim, backend="jaxfne")

        # Both should have same structure
        assert len(legacy.trials) == len(jaxfne_res.trials)
        assert set(legacy.trials[0].keys()) == set(jaxfne_res.trials[0].keys())

        # Shapes should match
        for area in ["V1"]:
            legacy_area = legacy.trials[0][area]
            jaxfne_area = jaxfne_res.trials[0][area]

            assert legacy_area["spikes"].shape == jaxfne_area["spikes"].shape
            assert legacy_area["voltage_mV"].shape == jaxfne_area["voltage_mV"].shape
            assert legacy_area["lfp_contacts"].shape == jaxfne_area["lfp_contacts"].shape


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
