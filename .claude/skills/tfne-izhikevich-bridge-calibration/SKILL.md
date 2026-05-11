# Skill: TFNE-Izhikevich Bridge Calibration

**Purpose:** Guide work on converting Izhikevich point-neuron outputs into TFNE-safe source terms.

**Use when:** Editing `src/jbiophysic/models/tfne_izhikevich.py`, tutorials using Izhikevich-to-TFNE, or examples that map spikes/currents to fields.

## Required Doctrine
- Izhikevich `I` is native/current-like.
- Physical amperes require explicit positive scale.
- Uncalibrated source traces may support timing/spike demonstrations, but NOT physical LFP/CSD amplitude claims.

## Required Gates
- **Positive Calibration:** Verify a positive calibration scale is used for physical units.
- **Refusal for Uncalibrated Claims:** Refuse to make amplitude claims for uncalibrated models; include metadata status.
- **Unit Declaration:** Explicitly declare source-current units.
- **Finite Values:** Ensure no NaN or Inf in bridge outputs.
- **Nonzero Spiking:** Smoke models must maintain nonzero spiking when expected.
- **Clean Environment:** Generated outputs must be untracked.

## Validation Commands
- `python -m compileall -q src tests examples tutorials`
- `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=src python -m pytest -q tests/test_tfne_sources.py tests/test_tfne_tensors_fields.py tests/test_tfne_p0_invariants.py tests/models/test_lap_izhikevich_baseline.py`

## Stop Conditions
- Biological-proof language.
- Source amplitude claim without explicit calibration.
- Source conservation not checked.
- Changed emitter dynamics without dedicated Izhikevich validation.
