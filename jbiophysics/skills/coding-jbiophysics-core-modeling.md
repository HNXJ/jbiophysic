---
name: coding-jbiophysics-core-modeling
description: Hierarchical biophysical modeling and optimization API using Jaxley, NetBuilder, and OptimizerFacade.
---
# coding-jbiophysics-core-modeling

This skill documents the unified API for building, simulating, and optimizing hierarchical biophysical networks.

## 1. Multi-Area Fluent Builder: `NetBuilder`
Declarative construction of hierarchical circuits using the `Area.Population` indexing pattern.

### Key Logic & API
- **Area Indexing**: `builder.add_population(..., area="V1")` creates isolated logical blocks.
- **Param Independence**: `builder.make_trainable(["gAMPA"])` makes every `gAMPA` in the network independent for tuning.
- **Traceability**: `builder.build()` returns a flattened `jx.Network` while preserving area metadata in `builder.population_offsets`.

## 2. Differentiable Optimization: `OptimizerFacade`
A unified interface for **AGSDR v2** (Adaptive Genetic-Stochastic Delta-Rule) with an **Adam** inner optimizer.

### Highly Useful Hints
- **Constraint Weights**: Use `set_constraints(firing_rate=(1, 50), weight=1.0)` to balance stability vs. precision.
- **Population Locking**: Set per-area constraints (e.g., `V1.E`) to prevent higher areas from overpowering sensory input during Joint-Stage tuning.
- **Alpha Floor**: The `0.1` floor in AGSDR is critical; if the network stops exploring, check if `var_stochastic` has collapsed.
- **Squared Hinge Loss**: Replaces exponential penalties with `(jnp.maximum(0, val - limit)**2)`. This ensures numerical stability at extreme parameter values.

## 3. Highly Useful Hints for Implementation
- **SafeHH Naming**: Always use `SafeHH(name="HH")`. `Jaxley` maps parameters by name; inconsistent naming leads to untrainable parameters.
- **Broadcasting Stimuli**: Stimuli MUST be 1D `(T,)`. Jaxley automatically broadcasts this across all compartments in a `cell_view`.
- **Differentiable Control Flow**: Use `jax.lax.cond` or `jnp.where`. Never use Python `if` inside a function that will be `jax.jit` compiled (like the loss function).

## 4. Common Pitfalls
- **Python Control Flow**: Using `if val > 0:` inside a loss function will cause a `ConcretizationError`.
- **NaN Gradients**: Often caused by `jnp.exp(v)` in the HH equations. `SafeHH` clamps `v` to `[-100, 100]` to mitigate this.
- **Unlinked Parameters**: Forgetting `make_trainable()` will result in zero gradients for those parameters.
