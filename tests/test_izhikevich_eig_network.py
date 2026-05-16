import json

import numpy as np

from jbiophysic.networks.izhikevich_eig import net_eig, simulate_eig_izhikevich


def test_net_eig_izhikevich_preserves_population_groups_and_params():
    net = net_eig(8, 2, 3, seed=7)
    assert net.n_neurons == 13
    assert net.counts_by_group() == {"E": 8, "Ing": 2, "Inl": 3}
    assert np.all(net.radius == 1.0)
    assert np.all(net.length == 1.0)

    a = net.izhikevich["a"]
    b = net.izhikevich["b"]
    d = net.izhikevich["d"]
    assert np.allclose(a[net.group_index["E"]], 0.02)
    assert np.allclose(b[net.group_index["Ing"]], 0.25)  # SST-like LTS
    assert np.allclose(a[net.group_index["Inl"]], 0.10)  # PV-like FS
    assert np.allclose(d[net.group_index["Inl"]], 2.0)
    assert net.source_calibration_status == "uncalibrated_izhikevich_native_current"


def test_net_eig_izhikevich_preserves_supplied_connectivity_motif():
    num_e, num_ig, num_il = 10, 4, 5
    n = num_e + num_ig + num_il
    net = net_eig(num_e, num_ig, num_il)
    specs = net.synapses

    # E -> all AMPA, Ing -> all slow GABAa, Inl -> 10% selected posts fast GABAa.
    assert specs[0].receptor == "AMPA"
    assert specs[0].tauD_ms == 2.0
    assert specs[0].pre.size == num_e * n
    assert specs[1].receptor == "GABAa"
    assert specs[1].tauD_ms == 5.0
    assert specs[1].pre.size == num_ig * n
    assert specs[2].receptor == "GABAa"
    assert specs[2].tauD_ms == 2.0
    assert specs[2].pre.size == num_il * int(n * 0.1)
    assert specs[0].pre_branch == 0 and specs[0].post_branch == 1
    assert specs[0].pre_loc == 0.0 and specs[0].post_loc == 1.0


def test_net_eig_izhikevich_json_and_smoke_simulation_are_finite():
    net = net_eig(6, 2, 2, seed=3)
    json.dumps(net.to_json_dict(), allow_nan=False)

    drive = np.zeros((80, net.n_neurons), dtype=np.float32)
    drive[:, net.group_index["E"]] = 8.0
    result = simulate_eig_izhikevich(net, drive, dt_ms=0.5, seed=4)
    assert result["V_mV"].shape == (80, net.n_neurons)
    assert result["spikes"].shape == (80, net.n_neurons)
    assert result["I_syn_native"].shape == (80, net.n_neurons)
    assert bool(result["finite"])
    assert np.all(np.isfinite(result["firing_rate_hz"]))
