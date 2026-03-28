# jbiophys
Hierarchical, JAX-differentiable biophysical modeling and optimization using **Jaxley** and **AGSDR**.

## 1. Building a Hierarchy
The `NetBuilder` provides a fluent API for constructing multi-area cortical circuits with area-aware indexing.

```python
from jbiophysics.compose import NetBuilder

# 1. Build a 2-area hierarchy (V1 + HO)
builder = (NetBuilder(seed=42)
    .add_population("E", n=80, cell_type="pyr", area="V1")
    .add_population("PV", n=20, cell_type="pv", area="V1")
    .add_population("E", n=50, cell_type="pyr", area="HO")
    # Intra-areal connection
    .connect("E", "PV", synapse="AMPA", p=0.1, area="V1")
    # Inter-areal Feedforward (V1 -> HO)
    .connect("V1.E", "HO.E", synapse="AMPA", p=0.05, g=0.5)
    .make_trainable(["gAMPA", "gGABAa"]))

net = builder.build()
```

## 2. Simulation
Integration is fully differentiable via **Jaxley**.

```python
import jaxley as jx
import numpy as np

# Record somatic voltage for all cells
net.delete_recordings()
net.cell("all").branch(0).loc(0.0).record("v")

# Integrate (40 kHz sampling)
traces = jx.integrate(net, delta_t=0.025, t_max=1000.0)
traces_np = np.array(traces) # (N_cells, T_steps)
```

## 3. Optimization with AGSDR
The `OptimizerFacade` orchestrates high-dimensional parameter tuning using the **Adaptive Genetic-Stochastic Delta-Rule (AGSDR v2)** with an **Adam** inner optimizer.

```python
from jbiophysics.compose import OptimizerFacade

# 1. Initialize Optimizer (AGSDR + Adam)
facade = (OptimizerFacade(net, method="AGSDR", lr=1e-3)
    .set_pop_offsets(builder.population_offsets)
    # 2. Set Multi-Objective Constraints
    .set_constraints(firing_rate=(1.0, 50.0), kappa_max=0.1) # Global
    .set_pop_constraints("V1.E", firing_rate=(2.0, 10.0))    # Per-area
    # 3. Run Training Loop
    .run(epochs=100, dt=0.025, t_max=1000.0))

# 4. Access Optimized Parameters and Report
report = facade
print(f"Final Loss: {report.metadata['history']['loss'][-1]}")
```

## Project Structure
All core logic, scripts, and research plans are consolidated within the `jbiophysics/` package directory:

- **jbiophysics/core/**: Biophysical primitives (mechanisms, neurons, optimizers).
- **jbiophysics/systems/**: Network architectures and simulation pipelines.
- **jbiophysics/scripts/**: Trial execution and batch processing.
- **jbiophysics/plans/**: Markdown research plans for systematic experimentation.
- **jbiophysics/results/**: Generated data, reports, and dashboards.
- **jbiophysics/viz/**: Visualization and analysis tools.
- **jbiophysics/functions/**: Reusable signal processing utilities.
- **jbiophysics/skills/**: Expert agent skills.

## Installation
```bash
pip install -e .
```
