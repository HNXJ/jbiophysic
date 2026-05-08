import numpy as np

from jbiophysic.models.laminar_oddball import (
    build_three_area_cortex,
    drive_schedule,
    global_oddball_sequences,
    lichtenfeld_three_layer_priors,
    omission_sequences,
    simulate_laminar_izhikevich,
    summarize_simulation,
)


def test_lichtenfeld_three_layer_priors_are_positive_simplexes():
    priors = lichtenfeld_three_layer_priors()
    assert set(priors) == {"low", "mid", "high"}
    for arr in priors.values():
        assert arr.shape == (3, 4)
        assert np.all(arr > 0)
        assert np.allclose(arr.sum(axis=1), 100.0)
    assert priors["high"][0, 3] > priors["low"][0, 3]  # CR/VIP increases up hierarchy.
    assert priors["low"][1, 1] > priors["high"][1, 1]  # PV middle-layer prior decreases.


def test_three_area_cortex_structure_matches_requested_constraints():
    cortex = build_three_area_cortex(n_neurons=300, seed=3)
    assert cortex.n_neurons == 300
    assert cortex.counts_by_area_layer_type().shape == (3, 3, 4)
    assert np.all(cortex.counts_by_area_layer_type() > 0)
    assert np.all(cortex.counts_by_area_layer_type().sum(axis=(1, 2)) == 100)
    assert np.all(cortex.counts_by_area_layer_type().sum(axis=2) == np.asarray([45, 20, 35]))
    assert cortex.edges["source"].size > 0
    assert set(cortex.edges["receptor_names"].tolist()) == {"AMPA", "GABA", "NMDA"}
    classes = set(cortex.edges["connection_class"].tolist())
    assert "feedforward_superficial_to_middle" in classes
    assert "feedback_superficial_deep_to_deep" in classes
    assert "apical_sst_to_half_deep_e" in classes


def test_tasks_share_timing_and_drive_omission_has_no_p4_input():
    cortex = build_three_area_cortex(n_neurons=300, duration_ms=1000.0, seed=4)
    go = global_oddball_sequences()
    om = omission_sequences()
    assert len(go["global_oddball"]) == len(om["omit_P4_A"]) == 4
    # Shortened duration clips later slots but keeps timing object common.
    full_cortex = build_three_area_cortex(n_neurons=300, duration_ms=5000.0, seed=4)
    standard = drive_schedule(full_cortex, "AAAA", amplitude=5.0, background=0.0)
    omitted = drive_schedule(full_cortex, "AAAX", amplitude=5.0, background=0.0)
    _name, p4_t0, p4_t1 = full_cortex.timing.event_slots[3]
    i0 = int(p4_t0 / full_cortex.dt_ms)
    i1 = int(p4_t1 / full_cortex.dt_ms)
    lower_mid_e = np.flatnonzero(
        (full_cortex.area_index == 0)
        & (full_cortex.layer_index == 1)
        & (full_cortex.cell_type_index == 0)
    )
    assert standard[i0:i1, lower_mid_e].sum() > 0
    assert omitted[i0:i1, lower_mid_e].sum() == 0


def test_smoke_simulation_is_finite():
    cortex = build_three_area_cortex(n_neurons=300, duration_ms=80.0, dt_ms=0.1, seed=5)
    drive = drive_schedule(cortex, "AAAA", amplitude=3.0, background=1.5)
    result = simulate_laminar_izhikevich(cortex, drive, seed=6, noise_sd=0.1, plasticity_enabled=True)
    summary = summarize_simulation(result, cortex.dt_ms)
    assert summary["finite"]
    assert np.isfinite(summary["mean_rate_hz"])
    assert result["V_mV"].shape == (800, 300)
    assert result["spikes"].shape == (800, 300)
    assert np.all(result["final_weights"] >= 0)
    assert np.all(result["final_weights"] <= 0.25)


def test_replication_manifest_and_constraint_audit_capture_scaffold():
    from jbiophysic.models.laminar_oddball import (
        density_priors_table,
        replication_manifest,
        validate_replication_constraints,
    )

    cortex = build_three_area_cortex(n_neurons=300, seed=10)
    table = density_priors_table()
    assert len(table) == 9
    assert {row["area"] for row in table} == {"low", "mid", "high"}
    audit = validate_replication_constraints(cortex)
    assert audit["n_neurons"] == 300
    assert audit["three_areas"]
    assert audit["no_zero_area_layer_type_bins"]
    assert audit["interarea_connections_are_excitatory"]
    assert 0.4 <= audit["apical_targets_fraction_of_deep_e"] <= 0.6
    manifest = replication_manifest(cortex)
    assert manifest["truth_status"] == "tutorial_exploratory_not_biological_truth"
    assert manifest["tasks"]["global_oddball"]["global_oddball"] == "AAAA"


def test_edge_masks_and_perturbations_support_causal_ablations():
    from jbiophysic.models.laminar_oddball import edge_mask, perturb_cortex_edges

    cortex = build_three_area_cortex(n_neurons=300, seed=11)
    pv_gaba = edge_mask(cortex, source_type="PV", receptor="GABA")
    assert pv_gaba.sum() > 0
    silenced = perturb_cortex_edges(cortex, source_type="PV", receptor="GABA", scale=0.0)
    assert np.all(silenced.edges["weight"][pv_gaba] == 0.0)
    assert np.any(cortex.edges["weight"][pv_gaba] > 0.0)

    feedback = edge_mask(cortex, connection_class="feedback_superficial_deep_to_deep")
    no_feedback = perturb_cortex_edges(
        cortex, connection_class="feedback_superficial_deep_to_deep", scale=0.0
    )
    assert feedback.sum() > 0
    assert np.all(no_feedback.edges["weight"][feedback] == 0.0)


def test_condition_batch_objectives_and_tfne_source_proxy_are_finite():
    from jbiophysic.models.laminar_oddball import (
        build_drive_batch,
        global_oddball_sequences,
        oddball_objectives,
        omission_objectives,
        omission_sequences,
        population_activity_proxy,
        simulate_condition_batch,
        tfne_source_proxy,
    )

    cortex = build_three_area_cortex(n_neurons=300, duration_ms=100.0, dt_ms=0.1, seed=12)
    # Short smoke drives are clipped before P4, but still validate API and finite outputs.
    go_drives = build_drive_batch(cortex, global_oddball_sequences(), amplitude=3.0, background=1.0)
    go_results = simulate_condition_batch(cortex, go_drives, seed=13, noise_sd=0.0, plasticity_enabled=False)
    go_obj = oddball_objectives(go_results, cortex)
    assert set(go_results) == set(global_oddball_sequences())
    assert all(np.isfinite(v) for v in go_obj.values())

    om_drives = build_drive_batch(cortex, omission_sequences(), amplitude=3.0, background=1.0)
    om_subset = {k: om_drives[k] for k in ["standard_A", "omit_P4_A"]}
    om_results = simulate_condition_batch(cortex, om_subset, seed=17, noise_sd=0.0, plasticity_enabled=False)
    om_obj = omission_objectives(om_results, cortex)
    assert all(np.isfinite(v) for v in om_obj.values())

    proxy = population_activity_proxy(next(iter(go_results.values())), cortex, group_by="area")
    assert set(proxy) == {"low", "mid", "high"}
    src = tfne_source_proxy(next(iter(go_results.values())), cortex, spike_current_A=1e-12)
    assert src["positions_m"].shape == (300, 3)
    assert src["currents_A"].shape == (1000, 300)
    assert np.isfinite(src["currents_A"]).all()
