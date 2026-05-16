# Final Pre-Simulation Validation Report

Date: 2026-05-14
Package: `jbiophysic` extracted from `jbiophysic-main.zip`
Version set: `1.0.2`
Truth mode: `truth_safe_unverified`
Decision: `READY_FOR_SIMULATION_AND_TESTS`

## Scope

This bundle closes the pre-simulation tooling phase. It is a final, merge-ready computational scaffold for TFNE/JTFNE spectrolaminar and omission model-lite work. It does not claim biological proof, calibrated CSD/LFP amplitude, or a validated omission mechanism.

## Implemented hardening

1. Removed core import dependence on optional Optax.
2. Preserved Optax-backed optimizer APIs as optional `[jax]` functionality.
3. Added formal TFNE operator-status export keyed by manuscript symbols:
   `E_theta`, `S_WDR`, `C_mu_nu`, `Q_eta_alpha`, `F_field`, `P_probe`, `A_objective`, `O_optimizer`, `C_constraints`.
4. Added required field invariant metadata to JTFNE basis/manifests:
   source decomposition, source calibration, source projection mode, Neumann compatibility, gauge, boundary, solver status, CSD sign convention, finite flags, conductivity diagnostics, and residual fields.
5. Hardened deterministic spectrolaminar CLI outputs with null, ablation, and paired-seed evidence tables.
6. Added omission model-lite config and CLI as the settled R5 execution surface.
7. Kept claims conservative: smoke and scaffold outputs are not biological mechanism evidence.

## Changed / added files

- `README.md`
- `pyproject.toml`
- `configs/omission_model_lite.yaml`
- `scripts/run_spectrolaminar_suite.py`
- `scripts/run_omission_model_lite.py`
- `src/jbiophysic/jtfne.py`
- `src/jbiophysic/models/omission_lite.py`
- `src/jbiophysic/optim/__init__.py`
- `src/jbiophysic/optim/agsdr.py`
- `src/jbiophysic/optim/gsdr.py`
- `src/jbiophysic/optim/gsgd.py`
- `src/jbiophysic/optim/sdr.py`
- `src/jbiophysic/tfne/operator_status.py`
- `tests/optim/test_agsdr_optax.py`
- `tests/optim/test_gsdr_optax.py`
- `tests/optim/test_gsgd_optax.py`
- `tests/optim/test_optim_jit_pmap.py`
- `tests/optim/test_sdr_optax.py`
- `docs/audit/FINAL_PRE_SIMULATION_VALIDATION_REPORT.md`

## Validation commands and observed results

### Compile

```bash
PYTHONPATH=src python -m compileall -q src tests examples scripts
```

Observed: pass.

### Targeted TFNE/JTFNE/core evidence tests

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=src python -m pytest -q \
  tests/test_import_smoke.py tests/test_optim_network_pipeline.py tests/jtfne tests/tfne \
  tests/common tests/configs tests/objectives tests/analysis \
  tests/test_tfne_sources.py tests/test_tfne_tensors_fields.py tests/test_tfne_p0_invariants.py
```

Observed: `34 passed in 20.05s`.

### Broader core/model/viz tests

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=src python -m pytest -q \
  tests/core tests/data tests/models tests/smoke tests/test_analysis.py tests/test_cells.py \
  tests/test_cortex_network_builder.py tests/test_edge_backend.py tests/test_import_smoke.py \
  tests/test_objectives.py tests/test_simplified_api_imports.py \
  tests/test_simplified_api_ops.py tests/test_viz_jvis.py tests/viz
```

Observed: `62 passed, 1 skipped, 2 warnings in 21.36s`.

Warnings are SciPy Welch/spectrogram warnings for short test traces in `viz/jvis.py`; they are expected smoke-test warnings.

### Test collection

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=src python -m pytest --collect-only -q
```

Observed: `93 tests collected in 3.39s`.

Note: the monolithic aggregate pytest command printed pass summaries in this container but did not always terminate cleanly after JAX/background-thread teardown. Chunked validation is therefore the recommended acceptance path in constrained worker environments.

### Spectrolaminar deterministic CLI smoke

```bash
PYTHONPATH=src python scripts/run_spectrolaminar_suite.py \
  --config configs/spectrolaminar_v1.yaml \
  --seed 0 \
  --out outputs/spectrolaminar_v1_seed0 \
  --smoke
```

Observed: pass. Required outputs were generated:

- `manifest.json`
- `metrics.csv`
- `celltype_diagnostics.csv`
- `area_diagnostics.csv`
- `synchrony_diagnostics.csv`
- `field_invariants.csv`
- `operator_status.json`
- `null_metrics.csv`
- `ablation_metrics.csv`
- `paired_seed_table.csv`
- `asset_hashes.json`
- `figures/`

`python -m json.tool` passed on `manifest.json` and `operator_status.json`.

### Omission model-lite CLI smoke

```bash
PYTHONPATH=src python scripts/run_omission_model_lite.py \
  --config configs/omission_model_lite.yaml \
  --seeds 0 \
  --out outputs/omission_model_lite_v1 \
  --smoke
```

Observed: pass. Required outputs were generated:

- `manifest.json`
- `condition_metrics.csv`
- `spike_diagnostics.csv`
- `field_invariants.csv`
- `bandpower_by_condition.csv`
- `synchrony_diagnostics.csv`
- `ablation_metrics.csv`
- `null_metrics.csv`
- `parameter_bounds.csv`
- `asset_hashes.json`

`python -m json.tool` passed on `manifest.json`.

## Claim discipline

Allowed:

> The repository is ready for the simulation/test phase with deterministic evidence surfaces for spectrolaminar scaffolds and omission model-lite smoke runs.

Forbidden:

> The smoke outputs prove a biological omission mechanism.

> Izhikevich native current is calibrated physical current.

> Spectrolaminar ratio smoke proves E/I-ratio necessity.

## Final decision

`READY_FOR_SIMULATION_AND_TESTS`

The next work should be neuroscience execution: multi-seed omission model-lite runs, nulls, ablations, parameter-bound diagnostics, synchrony rejection, and empirical comparison. No further tooling churn is recommended unless a failing simulation/test exposes a blocking defect.
