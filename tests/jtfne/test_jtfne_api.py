from pathlib import Path

import numpy as np

from jbiophysic import jtfne


def test_cfg_save_load_roundtrip(tmp_path):
    cfg = jtfne.default_cfg("correct", smoke=True)
    path = tmp_path / "cfg.yaml"
    jtfne.save_cfg(cfg, path)
    loaded = jtfne.load_cfg(path)
    assert loaded.init.mode == "correct"
    assert loaded.sim.t_ms <= cfg.sim.t_ms
    assert loaded.init.n_neuron_per_column == cfg.init.n_neuron_per_column


def test_construct_correct_and_inverse_have_expected_deep_e_fraction():
    correct = jtfne.construct(jtfne.default_cfg("correct", smoke=True).init)
    inverse = jtfne.construct(jtfne.default_cfg("inverse", smoke=True).init)
    c_deep = correct.neurons[correct.neurons.layer.isin(["L5", "L6"])].cell_type.eq("E").mean()
    c_sup = correct.neurons[correct.neurons.layer.isin(["L1", "L2", "L3"])].cell_type.eq("E").mean()
    i_deep = inverse.neurons[inverse.neurons.layer.isin(["L5", "L6"])].cell_type.eq("E").mean()
    i_sup = inverse.neurons[inverse.neurons.layer.isin(["L1", "L2", "L3"])].cell_type.eq("E").mean()
    assert c_deep > c_sup
    assert i_deep < i_sup


def test_model_save_load_roundtrip(tmp_path):
    model = jtfne.construct(jtfne.default_cfg("correct", smoke=True).init)
    path = tmp_path / "model.pkl"
    jtfne.save_model(model, path)
    loaded = jtfne.load_model(path)
    assert len(loaded.neurons) == len(model.neurons)
    assert loaded.config_init.mode == "correct"


def test_simulate_evaluate_manifest_smoke(tmp_path):
    cfg = jtfne.default_cfg("correct", smoke=True)
    model = jtfne.construct(cfg.init)
    signals = jtfne.simulate(model, cfg.sim)
    assert len(signals.trials) == cfg.sim.n_trials
    for area in cfg.init.area_order:
        meta = signals.trials[0][area]["metadata"]
        assert meta["source_projection"] == "source_sink_return_current"
        assert meta["basis_conservation_max_abs"] < 1e-4
        assert np.isfinite(meta["solver_residual_max"])
    ev = jtfne.evaluate(signals, cfg.opt)
    assert ev.mean_similarity >= 0.0
    assert set(ev.scores.area) == set(cfg.init.area_order)
    manifest = jtfne.write_manifest(signals, tmp_path, evaluation=ev)
    assert manifest["truth_mode"] == "truth_safe_unverified"
    assert Path(tmp_path / "manifest.json").exists()


def test_correct_ratio_scores_above_inverse_in_smoke():
    correct_cfg = jtfne.default_cfg("correct", smoke=True)
    inverse_cfg = jtfne.default_cfg("inverse", smoke=True)
    correct = jtfne.construct(correct_cfg.init)
    inverse = jtfne.construct(inverse_cfg.init)
    correct_eval = jtfne.evaluate(correct, correct_cfg.opt)
    inverse_eval = jtfne.evaluate(inverse, inverse_cfg.opt)
    # This is a scaffold-level test: correct deep-high E/I should be better for the
    # declared deep alpha/beta and superficial gamma target than the inverse control.
    assert correct_eval.mean_similarity > inverse_eval.mean_similarity


def test_optimize_runs_declared_sweep_without_mutating_claim_level():
    cfg = jtfne.default_cfg("correct", smoke=True)
    model = jtfne.construct(cfg.init)
    optimized = jtfne.optimize(model, cfg.opt)
    assert hasattr(optimized, "optimization")
    assert len(optimized.optimization.optimization_log) >= 1
    assert optimized.truth_mode == "truth_safe_unverified"
