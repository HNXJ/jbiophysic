# jaxfne Neural-Circuit Playground

Status: `truth_safe_unverified`, `computational_scaffold`, `laminar_proxy_no_pde`.

This layer makes `jbiophysic` a safe playground for jaxfne-based neural circuit network models. It is a thin adapter around the jaxfne engine, not a second simulator. Runtime wrappers use the canonical engine import:

```python
import jaxfne as jtfne
```

## What the playground provides

- dependency-safe request manifests when jaxfne is not installed;
- lazy executable smoke runs when jaxfne is installed;
- stable wrappers for jaxfne suite configs, construction, simulation, and signal-contract validation;
- explicit status gates for proxy field outputs and source calibration status;
- JSON receipts suitable for worker reports.

## Supported jaxfne suites

```text
suite2_single_neuron
suite2_four_celltype
suite2_net1
suite2_v1_v4
```

These map to jaxfne public builders:

```text
suite2_single_neuron_config
suite2_four_celltype_config
suite2_net1_config
suite2_v1_v4_config
```

Then the adapter calls:

```text
jtfne.construct(cfg)
jtfne.simulate(model, duration_ms=..., dt_ms=..., seed=...)
```

## Dry-run manifest

No jaxfne install is required:

```bash
PYTHONPATH=src python examples/jaxfne_playground_smoke.py \
  --name suite2_four_celltype \
  --seed 0 \
  --duration-ms 10 \
  --dt-ms 0.1 \
  --out outputs/jaxfne_playground_smoke
```

Expected output:

```text
outputs/jaxfne_playground_smoke/jaxfne_playground_request_manifest.json
```

## Executable smoke

Requires jaxfne and JAX extras:

```bash
pip install -e ".[jax,jaxfne]"
PYTHONPATH=src python examples/jaxfne_playground_smoke.py \
  --name suite2_four_celltype \
  --seed 0 \
  --duration-ms 10 \
  --dt-ms 0.1 \
  --out outputs/jaxfne_playground_smoke \
  --execute
```

The executable smoke validates the corrected selector/signal contract:

```python
cfg = jtfne.suite2_four_celltype_config(seed=0, duration_ms=10.0, dt_ms=0.1)
model = jtfne.construct(cfg)
idx = model.select(cell_type="E")
signals = jtfne.simulate(model, duration_ms=10.0, dt_ms=0.1, seed=0)
assert signals.get("vm").shape[-1] == int(signals.V_m.shape[-1])
assert signals.get("spk").shape[-1] == int(signals.V_m.shape[-1])
assert signals.get("vm", cell_type="E").shape[-1] == len(idx)
```

## Claim boundaries

Allowed wording:

```text
jaxfne-backed computational playground
source-to-field/readout scaffold
laminar proxy readout
uncalibrated spike/source proxy
```

Do not claim:

```text
real EEG/MEG
calibrated amplitude
metabolism
biological mechanism proof
PDE solve
```

Those claims require solver, geometry, calibration, boundary/gauge, residual, units, validation, nulls, and empirical comparison.
