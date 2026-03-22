# jbiophysics

Differentiable biophysical neural simulation using Jaxley + JAX. GSDR/AGSDR optimization, laminar columns, multi-area hierarchies.

## Quick Start
```python
import jbiophysics as jbp
net = (jbp.NetBuilder(seed=42)
    .add_population("E", n=80, cell_type="pyramidal")
    .add_population("I", n=20, cell_type="pv")
    .connect("E", "all", synapse="AMPA", p=0.1)
    .connect("I", "all", synapse="GABAa", p=0.4)
    .make_trainable(["gAMPA", "gGABAa"])
    .build())
```

## Modules
- `core/mechanisms/` — Inoise, GradedAMPA/GABAa/GABAb/NMDA
- `core/optimizers/` — SDR, GSDR, AGSDR
- `core/neurons/` — HH cell builders
- `systems/networks/` — LaminarColumn, InterArea
- `compose.py` — NetBuilder fluent API
- `export.py` — ResultsReport multi-format export
- `viz/` — Plotly raster, PSD, spectrogram, dashboard

## Install
```bash
pip install -e .
```
