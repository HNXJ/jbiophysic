# JTFNE smoke validation report

Status: `ACCEPT_CANDIDATE` for the notebook-facing `jtfne` API smoke bundle.

Archive basis: extracted `jbiophysic-main.zip`, updated locally with `src/jbiophysic/jtfne.py`, configs, tutorials, tests, and docs.

## Scope

This validation tests a developmental TFNE-Izhikevich spectrolaminar scaffold with two laminar E/I modes:

- `correct`: higher E/I in deep layers (L5/L6) and higher inhibitory fraction in superficial layers.
- `inverse`: inverse laminar ratio control.

The evaluated claim is bounded:

> Under this declared TFNE-Izhikevich source-to-field scaffold, the correct deep-high E/I laminar profile scores above the inverse control for the hand-declared target motif: deep alpha/beta and superficial gamma.

This is not biological proof, not empirical amplitude validation, and not a unique mechanism claim.

## Method/PDF alignment checks

Implemented in `jtfne`:

- Emitter: Izhikevich E/PV/SST/VIP nodes.
- Synaptic/recurrent state: directed local/feedforward/feedback weights and spike coupling.
- Source projection: calibrated native-current proxy to A via `source_scale_A_per_native`.
- Source compatibility: source-sink return-current kernels with Neumann-compatible zero net source.
- Tensor field: reduced isotropic extracellular field solve using existing TFNE smoke solver.
- Probe/readout: laminar LFP/CSD contact traces.
- Objective/evaluation: spectrolaminar similarity, firing-rate diagnostics, silent fraction, voltage range, synchrony kappa proxy.
- Optional optimizer: black-box sweep over declared gain/noise variables; not biological proof.

Not implemented as first-class operators in this bundle:

- explicit chemical modulation operator;
- electrodiffusion or true chemical source terms;
- calibrated EEG/MEG/head-model lead fields;
- full sparse/FEM production field solver.

## Commands run

```bash
PYTHONPATH=src python -m compileall -q src tests examples
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=src python -m pytest -q tests/jtfne/test_jtfne_api.py
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=src python -m pytest -q tests/test_tfne_p0_invariants.py tests/test_tfne_sources.py tests/test_tfne_tensors_fields.py tests/jtfne/test_jtfne_api.py
PYTHONPATH=src python examples/jtfne_spectrolaminar_smoke.py
PYTHONPATH=src python - <<'PY'
from jbiophysic import jtfne
print('jtfne import PASS', jtfne.default_cfg('correct', smoke=True).init.mode)
PY
```

## Results

- Compile: PASS.
- JTFNE tests: 6 passed.
- TFNE invariant + JTFNE targeted tests: 17 passed.
- Smoke example: PASS.
- Import smoke: PASS.

Smoke comparison from `examples/jtfne_spectrolaminar_smoke.py`:

```text
correct mean_similarity = 23.164168911542216
correct min_similarity  = 22.461236529422226
inverse mean_similarity = 14.359121987245345
inverse min_similarity  = 13.170537445854643
```

Decision: the correct deep-high E/I ratio mode scores above inverse in smoke mode under this scaffold.

## Notes

The full legacy test suite in this extracted archive still contains optional Optax-dependent optimizer tests. `tests/conftest.py` now skips optional Optax optimizer tests when Optax is unavailable in minimal CPU/Colab environments. The focused TFNE/JTFNE validation passed without requiring Optax.
