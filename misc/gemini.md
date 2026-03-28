# jbiophysics
Hierarchical, JAX-differentiable biophysical modeling and optimization.

## Core Architecture & Concepts
- **Hierarchical NetBuilder**: Fluent API for constructing multi-area circuits. Key concept: **Area-Aware Indexing** (`Area.Population`) enables granular control over connectivity and localized loss calculation.
- **Differentiable Integration**: Uses **Jaxley** for XLA-compiled ODE integration. **Mandate**: Stimulus inputs must be strictly 1D (`T,`) to maintain JAX traceability across single-compartment views.
- **Safe Biophysics**: Employs `SafeHH` primitives with voltage clamping and NaN guards. **Mandate**: Explicit naming (`name="HH"`) is required for consistent parameter mapping.
- **OptimizerFacade**: High-level orchestration of **AGSDR v2** (Adaptive Genetic-Stochastic Delta-Rule).
    - **Adaptive Alpha**: Balances gradient-based (supervised) and stochastic (unsupervised) updates using EMA-smoothed variance ratios.
    - **Stochastic Floor**: Maintains a hard `0.1` floor to prevent local minima deadlock (Stochastic Deadlock Prevention).
    - **Numerical Stability**: Replaces exponential penalties with **Squared Hinge Loss** to ensure convergence in high-dimensional landscapes.

## Calibration & Paradigm Workflow
- **Multi-Context Calibration**: Simultaneous optimization across BU/TD contexts (FF, Spontaneous, Attended, Omission) to find robust parameter regimes.
- **Two-Stage Tuning**:
    1. **Isolated Stage**: Calibration of local E/I balance and population synchrony (**Fleiss Kappa < 0.1**).
    2. **Joint Stage**: Calibration of inter-areal FF/FB motifs and spectral matching (**SSS via log-PSD MSE**).

## Folder Structure
- **core/**: Biophysical primitives (mechanisms, neurons, optimizers).
- **systems/**: Network architectures and high-level simulation pipelines.
- **functions/**: Reusable utility functions for signal processing and analysis.
- **scripts/**: Execution scripts for trials and batch processing.
- **plans/**: Markdown research plans for systematic experimentation.
- **skills/**: Expert agent skills for automated development and analysis.
- **results/**: Generated data, reports, and visualization artifacts.

## Global Workspace Rules
- **Folder Documentation**: Every folder under the workspace must contain a `README.md`.
- **Content Requirement**: Each `README.md` must have a minimum of 10 words for every file present in that folder's root (excluding the `README.md` itself).

## Quick Start
```python
import jbiophysics as jbp
net = (jbp.NetBuilder(seed=42)
    .add_population("E", n=80, cell_type="pyr", area="V1")
    .add_population("I", n=20, cell_type="pv", area="V1")
    .connect("E", "I", synapse="AMPA", p=0.1, area="V1")
    .make_trainable(["gAMPA"])
    .build())
```
