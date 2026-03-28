# jbiophys
Hierarchical, JAX-differentiable biophysical modeling and optimization.

## 1. Top-Level Architectural Mandates
- **Differentiability**: Zero Python control flow on traced arrays; use `jnp.where` and `jax.lax.cond`.
- **Area-Aware Indexing**: Use `Area.Population` (e.g., `V1.E`, `PFC.PV`) for all connectivity, metrics, and population-level loss calculation.
- **Traceability**: `NetBuilder` maintains a strict `N_cells` mapping. Stimulus inputs MUST be 1D `(T,)` for JAX-native broadcasting across the flattened network view.
- **Numerical Stability**: Use `SafeHH(name="HH")` for all cells. Replace exponential penalties with **Squared Hinge Loss** (`soft_range_loss`) for firing rate and synchrony (Kappa) constraints.

## 2. Advanced Optimization: AGSDR v2
- **Unified Engine**: `OptimizerFacade` orchestrates **Adaptive Genetic-Stochastic Delta-Rule**.
- **Adaptive Alpha**: `alpha = var_supervised / (var_supervised + var_stochastic)`. A hard `0.1` floor prevents "Stochastic Deadlock" by ensuring constant exploration.
- **Multi-Context Calibration**: Simultaneously optimize across multiple trial states (FF-only, Spontaneous, Attended, Omission) to find robust parameter regimes that generalize across sensory conditions.
- **Two-Stage Strategy**:
    1. **Stage 1 (Isolated)**: Local E/I balance and asynchrony targeting (**Fleiss Kappa < 0.1**).
    2. **Stage 2 (Joint)**: Inter-areal motif tuning and spectral matching (**log-PSD MSE**).

## 3. Highly Useful Hints
- **Seed Persistence**: Always initialize `NetBuilder(seed=X)` and `OptimizerFacade(seed=Y)` to ensure reproducible hierarchy and gradient trajectories.
- **Param Independence**: Use `.make_trainable(["gAMPA", "gGABAa"])` to decouple shared synaptic parameters for granular per-area tuning.
- **Laminar Flow**: $t=0$ should always align with the first sensory trigger (Code 101.0). Omission windows typically start at $t=1031ms$ (P2 onset).
- **Inhibition Control**: **PV+** deficits (perisomatic) primarily disrupt gamma/gain; **SST+** deficits (dendritic) impair subtractive prediction cancellation; **VIP+** cells act as "Omission Triggers" via disinhibition.

## 4. Folder Structure
- **core/**: Biophysical primitives (mechanisms, neurons, optimizers).
- **systems/**: Network architectures and high-level simulation pipelines.
- **functions/**: Reusable utility functions for signal processing (Morlet TFR, PSD).
- **scripts/**: Trial execution and batch processing scripts.
- **plans/**: Markdown research plans for systematic experimentation.
- **skills/**: Expert agent skills for automated development and analysis.
- **results/**: Generated data, reports, and visualization artifacts.

## 5. Workspace Rules
- **Folder Documentation**: Every folder must contain a `README.md`.
- **Content Requirement**: Minimum of 10 words for every file present in that folder's root (excluding the `README.md`).
- **Sync Protocol**: Push Python code to repos immediately after validation.
