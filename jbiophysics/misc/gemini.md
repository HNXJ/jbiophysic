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

## 4. Consolidated Structure
All assets are managed within the `jbiophysics/` package:
- **jbiophysics/core/**: Biophysical primitives (mechanisms, neurons, optimizers).
- **jbiophysics/systems/**: Network architectures and simulation pipelines.
- **jbiophysics/functions/**: Reusable signal processing (Morlet TFR, PSD).
- **jbiophysics/scripts/**: Trial execution and batch processing.
- **jbiophysics/plans/**: Markdown research plans for systematic experimentation.
- **jbiophysics/skills/**: Expert agent skills.
- **jbiophysics/results/**: Generated data, reports, and visualization artifacts.
- **jbiophysics/viz/**: Core visualization logic.
- **jbiophysics/misc/**: Metadata and global mandates.

## 5. Workspace Rules
- **Folder Documentation**: Every folder must contain a `README.md`.
- **Content Requirement**: Minimum of 10 words for every file present in that folder's root (excluding the `README.md`).
- **Sync Protocol**: Push Python code to repos immediately after validation.

## DIRECTIVE: Pre-Flight Context Gathering
When a user prompt involves data analysis, biophysical neuronal networks, or neurophysiological hypotheses (e.g., predictive routing, oscillation dynamics):
1. **HALT** immediate analytical generation.
2. **GATHER:** Autonomously use available tools (`grep_search`, `read_file`, shell commands) to search the local workspace for relevant GAMMA files, mathematical blueprints, or prior literature notes.
3. **PROCESS:** Only begin the final analysis once the extended theoretical and methodological context is fully loaded into active memory.

## DIRECTIVE: Autonomous Version Control
Whenever a script modification, code generation, or system configuration is executed and tested successfully without errors:
1. **PROMPT:** Immediately ask the user: *"Code execution successful. Shall I push these changes to the remote repository?"*
2. **PUSH:** If the user confirms, autonomously execute the standard Git flow (`git add .`, `git commit -m "update: [brief description of changes]"`, `git push`) via shell commands.
