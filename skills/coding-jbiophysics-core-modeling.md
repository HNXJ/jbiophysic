---
name: coding-jbiophysics-core-modeling
description: Hierarchical biophysical modeling and optimization API using Jaxley, NetBuilder, and OptimizerFacade.
---
# coding-jbiophysics-core-modeling

This skill documents the unified API for building, simulating, and optimizing hierarchical biophysical networks. It leverages **Jaxley** for differentiable simulations and provides a multi-area `NetBuilder`.

## 1. Multi-Area Fluent Builder: `NetBuilder`
Declarative construction of hierarchical circuits using the `Area.Population` indexing pattern.

### Example Construction
```python
import jbiophysics as jbp

builder = (jbp.NetBuilder(seed=42)
    .add_population("E", n=80, cell_type="pyr", area="V1")
    .add_population("PV", n=20, cell_type="pv", area="V1")
    .add_population("E", n=50, cell_type="pyr", area="HO")
    .connect("E", "PV", synapse="AMPA", p=0.1, area="V1")  # Intra-areal
    .connect("V1.E", "HO.E", synapse="AMPA", p=0.05)      # Feedforward
    .make_trainable(["gAMPA", "gGABAa"]))

net = builder.build()
```

## 2. Differentiable Optimization: `OptimizerFacade`
A unified interface for gradient-based (Adam) and stochastic (AGSDR) optimization.

### Adaptive GSDR (AGSDR v2) Logic
- **Alpha Scaling**: `alpha = var_supervised / (var_supervised + var_stochastic)`.
- **Stochastic Floor**: `alpha_min = 0.1` prevents deadlock by ensuring constant exploration.
- **Numerical Stability**: Uses **Squared Hinge Loss** (`soft_range_loss`) to prevent `jnp.exp` overflows.

### Multi-Objective Calibration
```python
facade = (jbp.OptimizerFacade(net, method="AGSDR", lr=1e-3)
    .set_pop_offsets(builder.population_offsets)
    .set_constraints(firing_rate=(1.0, 100.0), kappa_max=0.1) # Global
    .set_pop_constraints("V1.E", firing_rate=(2.0, 10.0))    # Per-area
    .set_target(target_psd_array))                           # Spectral

report = facade.run(epochs=100)
```

## 3. Metrics & Mechanisms
- **Fleiss Kappa**: Quantifies population synchrony; target **Kappa < 0.10** for asynchrony.
- **Spectral Match**: MSE on log-scale PSD to match oscillatory profiles (e.g., Alpha/Beta).
- **SafeHH**: Modified Hodgkin-Huxley with voltage clipping and NaN guards.
- **Graded Synapses**: `GradedAMPA`, `GradedGABAa`, `GradedNMDA`, `GradedGABAb`.

## 4. Parameter Management
- **Independence**: `make_trainable(["gAMPA"])` decouples shared parameters for granular tuning.
- **Offsets**: `builder.population_offsets` and `builder.area_offsets` provide the metadata required for per-area loss calculation.
