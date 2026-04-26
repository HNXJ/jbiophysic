# 🧠 jbiophysic
Research-grade biophysical simulation and analysis platform. (Axis 1-18)

## 🧬 Architecture: 3-Tier Layering
The repository follows a professional 3-tier architecture, implemented as a unified, pip-installable package (`jbiophysic`) with strict internal boundary enforcement.

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

### 📁 Resources (`assets/`)
- **Configs:** Experiment specifications and hyperparameters (`assets/configs/`).
- **Gamma:** AI-mediated trace results and metadata (`assets/gamma/`).

## 📜 Engineering Standards
- **Structured Logging:** Centralized logging replaces raw debug prints for library-grade transparency.
- **PyTree Discipline:** All models are `equinox.Module` subclasses, ensuring safe JAX transformations.
- **Tiered Testing:** Comprehensive validation across `core`, `models`, and `viz` tiers.
- **Root Hygiene:** Strict modularity; all implementation logic resides within `src/jbiophysic/`.

---
*Madelane Golden Dark (#CFB87C / #9400D3)*