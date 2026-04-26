# 🧠 jbiophysic
A JAX-native biophysical simulation and analysis platform designed for computational neuroscientists and biologists.

## 🧬 Architecture: 3-Tier Layering
The repository follows a clean 3-tier architecture, implemented as a unified, pip-installable package (`jbiophysic`) to separate mathematical mechanisms from scientific workflows and visualization.

### 1. Core Layer (`src/jbiophysic/core/`)
**Domain:** Computational Kernels & Biophysical Primitives.
- **Math:** Precision-weighted predictive coding and JAX-native differentiable math.
- **Mechanisms:** Ion channel kinetics (HH), synaptic ODEs, and calcium-modulated STDP.
- **Performance:** Hot paths are stripped of side-effects for optimal JIT compilation.

### 2. Models Layer (`src/jbiophysic/models/`)
**Domain:** Orchestration & Scientific Pipelines.
- **Builders:** Morphologically detailed cell populations and multi-area cortical hierarchies.
- **Simulation:** Orchestrated execution via `Equinox` modules and `Diffrax` solvers.
- **Optimization:** GSGD and AGSDR engines for biophysical parameter tuning.

### 3. Viz Layer (`src/jbiophysic/viz/`)
**Domain:** Visual Analytics & Data Persistence.
- **Serializers:** Decoupled payload generation for cross-platform visualization.
- **Scientific Protocol:** Strict JSON `null` protocol for `NaN/Inf` data integrity.
- **Aesthetic:** High-fidelity scientific plotting with a Madelane Golden Dark theme.

## 🛠 Installation & Usage
Install the package in development mode:
```bash
pip install -e ".[dev,viz]"
```

Execute a full scientific experiment pipeline via the CLI:
```bash
jbiophysic --run
```

Or run via the module path:
```bash
python -m jbiophysic.models.pipelines.run_full_experiment
```

### 📁 Resources
- **Configs:** Experiment specifications and hyperparameters (`configs/`).
- **Artifacts:** AI-mediated trace results and metadata (`artifacts/gamma/`).

## 📜 Engineering Standards
- **Structured Logging:** Centralized logging replaces raw debug prints for library-grade transparency.
- **PyTree Discipline:** All models are `equinox.Module` subclasses, ensuring safe JAX transformations.
- **Tiered Testing:** Comprehensive validation across `core`, `models`, and `viz` tiers.
- **Root Hygiene:** Strict modularity; all implementation logic resides within `src/jbiophysic/`.

---

## 🔬 Who this package is for

### Neurobiologist-friendly overview

`jbiophysic` is designed for researchers who want to work with biologically interpretable cortical circuits without dropping immediately into low-level simulator internals.

At a high level, the package lets you define excitatory and inhibitory populations, assemble them into local cortical columns or multi-area hierarchies, simulate membrane activity, extract spike rasters and voltage traces, and compare model activity against biologically meaningful targets such as firing rate, oscillatory band power, and excitation-inhibition balance.

The current codebase already includes:
- excitatory pyramidal-like cells and inhibitory interneuron subclasses,
- local column assembly with E / PV / SST / VIP populations,
- multi-area cortical hierarchy builders,
- simulation runners for voltage traces,
- spike-raster and trace serialization,
- spectral analysis utilities for beta and gamma bands,
- optimization components for fitting activity statistics.

This makes the package useful for questions such as:
- How does changing PV strength alter population spiking and oscillatory activity?
- How do SST and VIP pathways gate recurrent excitation?
- Can a simple cortical hierarchy reproduce a desired firing-rate regime?
- How does simulated activity compare to empirical beta/gamma structure?

### Computational neuroscience-friendly overview

`jbiophysic` is a JAX-native biophysical modeling stack organized into three layers:
- **core** for mechanisms, channels, synapses, and differentiable primitives,
- **models** for population builders, hierarchy construction, simulation orchestration, and optimization,
- **viz** for serialization and downstream plotting/analysis interfaces.

The current implementation includes:
- Hodgkin-Huxley-like channel logic,
- synaptic kinetics and modulation hooks,
- Jaxley-backed cell/network construction,
- simulation orchestration returning structured `SimulationResult`,
- spectral feature extraction from simulated signals,
- loss functions for rate, spectral targets, E/I balance, and stability,
- gradient-style and population-style optimization components.

This architecture is intended for users who want to move between:
- mechanistic neuron and synapse definitions,
- mesoscopic circuit assembly,
- differentiable or heuristic parameter fitting,
- reproducible export of activity into analysis and visualization pipelines.

---

## 🧩 Biological abstraction used in local circuits

The default local column builder uses four canonical cortical populations:

- **E / PC**: excitatory pyramidal cells
- **PV**: fast-spiking parvalbumin interneurons
- **SST**: somatostatin interneurons, often associated with dendritic inhibition
- **VIP**: vasoactive intestinal peptide interneurons, often associated with disinhibitory motifs

In the current code, these populations are assembled into a local cortical column and can then be combined into a larger multi-area hierarchy. This gives a natural starting point for studying recurrent excitation, inhibition-stabilized dynamics, disinhibition, and oscillatory regimes.

---

## ⚡ Current workflow in the codebase

The current stable workflow is builder-first rather than object-method-first.

```python
from jbiophysic.models.builders.hierarchy import build_cortical_hierarchy
from jbiophysic.models.simulation.run import run_simulation
from jbiophysic.common.types.simulation import SimulationConfig
from jbiophysic.viz import serialize_raster, serialize_voltage_traces

# 1. Build a small cortical hierarchy
brain = build_cortical_hierarchy(n_areas=2)

# 2. Simulate membrane activity
config = SimulationConfig(t_max=100.0, dt=0.05)
result = run_simulation(brain, config)

# 3. Convert voltage traces into spike-raster data
raster = serialize_raster(result, threshold=-20.0)

# 4. Extract traces for selected neurons
traces = serialize_voltage_traces(result, neuron_indices=[0, 1, 2])

print(f"Detected spikes: {len(raster.spike_times)}")
print(f"Trace duration: {raster.t_end} ms")
```

For spectral analysis:

```python
import numpy as np
from jbiophysic.models.observables.run_analysis import compute_spectral_features

# Example: analyze an LFP-like signal or other simulated time series
lfp = np.random.randn(1, 2000)
features = compute_spectral_features(lfp, fs=1000.0)

print(features["gamma_power"])
print(features["beta_power"])
print(features["peak_freq"])
```

---

## 🎯 Target ergonomic API

The long-term user-facing interface should be simpler and closer to how experimental and computational neuroscientists think about circuits.

A desirable high-level workflow would look like:

```python
from jbiophysic import network

N = network(E=200, pv=40, sst=40, vip=20)

sim = N.simulate(
    t_max=1000.0,
    dt=0.05,
    stimulus="oddball",
)

fit = N.train(
    target_rate_hz=5.0,
    target_gamma_power=0.4,
    target_beta_power=0.2,
)

figs = N.visualize(
    raster=True,
    traces=[0, 1, 2],
    spectrum=True,
)
```

This is the intended scientist-facing abstraction:

* `network(...)` defines the circuit in biological terms,
* `simulate(...)` runs activity and returns traces and raster-ready outputs,
* `train(...)` tunes parameters toward target activity statistics,
* `visualize(...)` provides standard circuit diagnostics.

At present, the repository exposes the underlying pieces of this workflow, but not yet this single consolidated interface.

---

## 🏛 Example: local cortical microcircuit

Conceptually, a simple local circuit can be described as:

```python
# planned ergonomic interface
N = network(E=200, pv=40, sst=40, vip=20)

result = N.simulate(t_max=500.0, dt=0.05)
N.visualize(raster=True, traces=[0, 10, 25])
```

Biologically, this corresponds to:

* recurrent excitatory drive among pyramidal cells,
* strong fast inhibition through PV cells,
* dendritic and contextual inhibition through SST cells,
* disinhibitory control through VIP cells.

This is a useful minimal motif for studying gain control, oscillatory balance, stimulus routing, and context-dependent circuit gating.

---

## 📈 Example: fitting a circuit to target activity

A common modeling workflow is not only to simulate, but to fit the model to desired activity statistics.

Conceptually, the interface should support:

```python
# planned ergonomic interface
fit = N.train(
    target_rate_hz=5.0,
    target_gamma_power=0.3,
    target_beta_power=0.1,
    target_ei_balance=1.0,
)
```

The current codebase already contains pieces of this objective:

* rate-based losses,
* beta/gamma spectral losses,
* E/I balance penalties,
* stability penalties,
* optimization modules for parameter updates.

Future training objectives will include synchrony metrics such as kappa-like population coordination. This means the scientific direction is already present, even though the final high-level training API is still being consolidated.

---

## 📊 Visualization outputs

The package is designed to support standard neuroscience readouts, including:

* spike rasters,
* membrane-potential traces,
* spectral summaries,
* network-level scene or connectivity payloads for downstream rendering.

In practical terms, the typical visual workflow is:

1. simulate voltage activity,
2. detect spikes from traces,
3. serialize raster and trace payloads,
4. render them in notebook, dashboard, or frontend tools.

---

## 🚀 Current status and near-term roadmap

The codebase already supports the main ingredients needed for a biologically grounded cortical workflow:

* cell and population builders,
* local column assembly,
* hierarchy construction,
* simulation,
* activity serialization,
* spectral analysis,
* optimization utilities.

The next ergonomic milestone is to expose these capabilities through a single user-facing object interface such as:

```python
# planned ergonomic interface
N = network(E=200, pv=40, sst=40, vip=20)
N.simulate(...)
N.train(...)
N.visualize(...)
```

That API would make the package substantially more accessible to both neurobiologists and computational neuroscientists while preserving the current layered implementation underneath.

---
*Madelane Golden Dark (#CFB87C / #9400D3)*
