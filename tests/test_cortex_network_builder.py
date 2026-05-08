import numpy as np

from jbiophysic.networks.cortex import DEFAULT_CELL_TYPES, make_cortex_network


def _example_network(**kwargs):
    return make_cortex_network(
        XYZ_mm=[0.5, 0.5, 1.5],
        N=120,
        Ls=[0.4, 0.2, 0.4],
        Ld=[
            [70, 10, 10, 10],
            [75, 20, 4, 1],
            [75, 15, 9, 1],
        ],
        seed=123,
        min_distance_um=8.0,
        max_connections=500,
        **kwargs,
    )


def test_cortex_network_counts_layers_and_types():
    net = _example_network(model_family="tfne-izhikevich", plasticity_coefficient=0.25)
    assert net.n_neurons == 120
    assert net.positions_mm.shape == (120, 3)
    assert net.positions_m.shape == (120, 3)
    counts = net.counts_by_layer_and_type()
    assert counts.shape == (3, 4)
    assert int(counts.sum()) == 120
    assert tuple(net.cell_types) == ("E", "PV", "SST", "VIP")


def test_cortex_network_positions_are_inside_layer_bounds_and_nonoverlap():
    net = _example_network(make_synapses=False)
    xyz = np.asarray(net.xyz_mm)
    assert np.all(net.positions_mm >= 0.0)
    assert np.all(net.positions_mm <= xyz)
    for layer_i, (z0, z1) in enumerate(net.layer_z_bounds_mm):
        z = net.positions_mm[net.layer_index == layer_i, 2]
        assert np.all(z >= z0)
        assert np.all(z <= z1)

    diffs = net.positions_mm[:, None, :] - net.positions_mm[None, :, :]
    dist = np.sqrt(np.sum(diffs * diffs, axis=-1))
    dist[dist == 0.0] = np.inf
    assert float(np.min(dist)) >= net.min_distance_mm - 1e-12


def test_cortex_network_synapse_metadata_uses_source_transmitter_rule():
    net = _example_network(model_family="izhikevich", plasticity_coefficient=1.5)
    src = net.synapses["source"]
    receptor_idx = net.synapses["receptor_index"]
    names = net.cell_type_names
    for pre, rec in zip(src, receptor_idx, strict=True):
        expected = 0 if names[int(pre)] == "E" else 1
        assert int(rec) == expected
    if len(src) > 0:
        assert np.allclose(net.synapses["plasticity_coefficient"], 1.5)


def test_cortex_network_tfne_hybrid_contains_explicit_calibration():
    net = _example_network(model_family="tfne-izhikevich", izh_current_to_ampere_scale=2e-12)
    assert net.tfne["source_positions_m"].shape == (120, 3)
    assert net.tfne["source_radii_m"].shape == (120,)
    assert net.tfne["izh_current_to_ampere_scale"] == 2e-12
    assert "CSD" in net.tfne["csd_sign_convention"]


def test_cell_type_specs_define_waveform_and_transmitter():
    assert DEFAULT_CELL_TYPES["E"].transmitter == "AMPA"
    assert DEFAULT_CELL_TYPES["PV"].transmitter == "GABA"
    assert DEFAULT_CELL_TYPES["SST"].waveform == "low_threshold_spiking"
    assert DEFAULT_CELL_TYPES["VIP"].polarity == "inhibitory"
