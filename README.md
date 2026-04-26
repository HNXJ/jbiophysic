# Gamma: jbiophysic
Research-grade biophysical simulation and analysis platform (Axis 1-10).

## 🧬 Architecture: 3-Tier Layering
The repository has been refactored into a pip-installable package (`jbiophysics`) with strict boundary enforcement.

### 1. Core Layer (`src/jbiophysic/core/`)
**Domain:** Pure Computation & Biophysical Kernels.
- **Math:** JAX-native differentiable math and predictive coding primitives.
- **Mechanisms:** Ion channel kinetics (HH), synapse ODEs, and STDP plasticity.
- **Solvers:** Integration with `diffrax` and custom JAX-compiled loops.

### 2. Models Layer (`src/jbiophysic/models/`)
**Domain:** Orchestration & Simulation Pipelines.
- **Builders:** Cell populations and inter-areal hierarchy construction via `Equinox` PyTrees.
- **Pipelines:** High-level experiment orchestration (e.g., Omission, Oddball).
- **Optimization:** GSGD and AGSDR biophysical tuning.

### 3. Viz Layer (`src/jbiophysic/viz/`)
**Domain:** Visual Analytics & Frontend Payloads.
- **Serializers:** Decoupled payload generation for web-agnostic visualization.
- **Plotly:** Professional-grade scientific plotting (Madelane Golden Dark theme).
- **Scientific Protocol:** Strict JSON `null` handling for NaNs/Infs to ensure data integrity.

## 🛠 Installation & Usage
Install in editable mode for research development:
```bash
pip install -e ".[dev,viz]"
```

Execute the manuscript generation CLI:
```bash
gravia-write --build
```

Run a full experiment pipeline:
```bash
python -m jbiophysic.models.pipelines.run_full_experiment
```

## 📜 Scientific Standards
- **Extreme Verbosity:** All execution traces are required to provide line-by-line transparency.
- **JAX Discipline:** All models are registered as `eqx.Module` to prevent JIT recompilation bloat.
- **Root Hygiene:** No new files allowed in the root. Use `src/` for code and `local/` for planning.
