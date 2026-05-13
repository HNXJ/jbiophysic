# Phase 2.2: TFNE-Izhikevich Spectrolaminar Notebook Integration Plan

**Status:** Audit Phase (No Implementation Yet)  
**Date:** 2026-05-13  
**Truth Mode:** truth_safe_unverified  
**Source:** TFNE-Izhikevich-Spectrolaminar-Motif-01-final-synaptic.ipynb / .py  

---

## Executive Summary

The attached notebook implements a **V1 → V4 → PFC TFNE-Izhikevich spectrolaminar motif scaffold** with:
- 200 neurons per cortical column, four cell classes (E/PV/SST/VIP)
- Laminar connectivity (local E/I, feedforward L2/3→L4, feedback L2/3/L6→L5/L6)
- TFNE source-to-field readout with basis precomputation
- Spectrolaminar similarity scoring and plasticity/gain/noise optimization
- Visualization suites (3D circuit, spectrolaminar profiles, activity rasters, CSD/LFP)

**Integration Challenge:** The notebook is a monolithic research artifact mixing model construction, TFNE readout, analysis, optimization, and visualization in one file. **Repo architecture requires modular separation.**

**Critical FieldSolution Compatibility Issue (Phase 2.1 Impact):**
- Notebook currently: `phi = jacobi_poisson_neumann_smoke(...)`
- After Phase 2.1: `solution = jacobi_poisson_neumann_smoke(...); phi = solution.phi_e`
- All TFNE basis building must preserve residual/status metadata from FieldSolution

**Decision Gate:** Before Phase 2.3 code movement, review: **Are proposed module boundaries correct?**

---

## Part A: Function Inventory

### A.1 Configuration & Setup (1 section, 70+ config values)

| Function | Location | Lines | Purpose |
|----------|----------|-------|---------|
| `config = SimpleNamespace(...)` | Cell/Section 1.b | ~100 | Centralized simulation config (anchors, anatomy, connectivity, TFNE, readout, viz) |
| `finalize_config(cfg)` | Cell 1.b | ~20 | Derive secondary params from anchors |
| `config_table = pd.DataFrame(...)` | Cell 1.b | ~10 | Summary display |

**Issues:**
- SimpleNamespace is mutable and type-unsafe
- 60+ parameters scattered in dict/tuple/list without validation
- Mixing simulation degrees of freedom (plasticity, gains) with readout objectives (alpha/beta/gamma bands)
- No manifest serialization path

**Proposed Target:** `src/jbiophysic/configs/spectrolaminar.py` + `configs/tfne_spectrolaminar_motif.yaml`

### A.2 Model Construction (9 functions, 250 lines)

| Function | Location | Lines | Purpose |
|----------|----------|-------|---------|
| `depth_to_l4_position(depth_m, cfg)` | Cell 2.a | ~2 | Map depth to L4-relative position |
| `params_for_cell(cell_type)` | Cell 2.a | ~5 | Return Izhikevich params by cell type |
| `integer_counts(total, fractions, keys)` | Cell 2.a | ~7 | Distribute integer counts via greedy allocation |
| `build_tfne_izhikevich_model(cfg)` | Cell 2.a | ~20 | Construct V1/V4/PFC 3-column network |
| `build_laminar_connections(neurons, positions_m, cfg)` | Cell 2.a | ~25 | Build connectivity (local E/I, FF, FB) |
| `connection_audit(model, cfg)` | Cell 2.a | ~6 | Summarize nonzero edges per connection type |
| `build_scaled_weight(model, control, cfg)` | Cell 2.a | ~4 | Apply control gains to connection matrices |
| (unused) | | | |

**Issues:**
- Model building returns dict, not structured dataclass
- No validation of cell counts, positions, connectivity density
- Connectivity is 4 separate matrices (no adjacency list or sparse format for visualization)
- "Synaptic resonance source" is mixed with connectivity, not separated as readout term

**Proposed Target:** `src/jbiophysic/models/spectrolaminar_motif.py`

### A.3 Emitter Simulation (4 functions, ~120 lines)

| Function | Location | Lines | Purpose |
|----------|----------|-------|---------|
| `simulate_emitters(model, control, seed, cfg)` | Cell 2.a | ~40 | Izhikevich dynamics + current + spike logic |
| `filtered_spike_source(spikes, neurons, control, seed, cfg)` | Cell 2.a | ~15 | Exponential filter spike state + cell signs |
| `synaptic_resonance_source(neurons, steps, control, seed, cfg)` | Cell 2.a | ~30 | Add fixed oscillatory bands (alpha, gamma) |
| `simulate_trials(model, control, n_trials, cfg, seed_offset)` | Cell 2.a | ~8 | Batch simulate_emitters with different seeds |

**Issues:**
- Spike state filtering and resonance source are hardcoded in one function
- Resonance source uses fixed frequencies and layer-dependent envelopes (strongly motif-shaped)
- No separation between "emitter output" and "TFNE source calibration"
- Resonance source is presented as spike-driven, not as a readout/scaffold term

**Critical Doctrine Issue:** The notebook says "spectrolaminar target is a readout objective, not an optimized generator." But `synaptic_resonance_source` produces alpha/gamma oscillations that match the target. This is intentional design, but the code must explicitly document it as a **readout scaffolding term**, not a biological mechanism proof.

**Proposed Target:** `src/jbiophysic/models/spectrolaminar_motif.py`

### A.4 TFNE Basis Building & Readout (2 functions, ~70 lines)

| Function | Location | Lines | Purpose |
|----------|----------|-------|---------|
| `build_tfne_basis(model, cfg)` | Cell 2.a | ~45 | Precompute LFP/CSD bases for each neuron in each area |
| `tfne_readout_trial(model, sim, cfg)` | Cell 2.a | ~20 | Project source currents to LFP/CSD contacts |

**FieldSolution Compatibility Issue (CRITICAL):**
```python
# Line 374 (current):
phi = jacobi_poisson_neumann_smoke(eta, grid, conductivity_s_m=cfg.TFNE_CONDUCTIVITY_S_M, steps=cfg.TFNE_JACOBI_STEPS)

# After Phase 2.1, must be:
solution = jacobi_poisson_neumann_smoke(eta, grid, conductivity_s_m=cfg.TFNE_CONDUCTIVITY_S_M, steps=cfg.TFNE_JACOBI_STEPS)
phi = solution.phi_e
# And preserve: residual_norm, n_iterations, converged, gauge_applied, boundary_condition
```

**Issues:**
- No capture of solver convergence status or residual
- No recording of gauge or boundary condition metadata
- Basis conservation error is computed but not integrated into structured dataclass
- readout trial returns dict; should return structured output

**Proposed Target:** `src/jbiophysic/models/tfne_spectrolaminar.py`

**Proposed New Dataclass:**
```python
@dataclass(frozen=True)
class TFNEReadoutBasis:
    lfp_basis: np.ndarray              # (n_neurons, n_contacts)
    csd_basis: np.ndarray              # (n_neurons, n_contacts)
    contact_depths_m: np.ndarray       # (n_contacts,)
    basis_conservation_max_abs: float
    solver_residual_norms: list[float] # One per neuron basis solve
    solver_converged_count: int        # How many converged vs max-iters
    solver_mean_iterations: float
    gauge_applied: str                 # "mean_zero"
    boundary_condition: str            # "neumann_zero"
    source_calibration_status: str     # "smoke_test"
```

### A.5 Spectrolaminar Analysis & Scoring (5 functions, ~80 lines)

| Function | Location | Lines | Purpose |
|----------|----------|-------|---------|
| `spectrolaminar_from_trials(trials, area, signal_key, cfg)` | Cell 2.a | ~25 | Compute FFT, interpolate to contacts, normalize to [0.48, 0.94] |
| `target_profiles(y, cfg)` | Cell 2.a | ~3 | Interpolate target alpha/beta and gamma at positions |
| `spectrolaminar_similarity(spec, cfg)` | Cell 2.a | ~20 | Multi-term error: shape MSE, anticorr penalty, L4 cross, layer drops |
| `summarize_similarity(trials, cfg)` | Cell 2.a | ~10 | Call spectrolaminar_from_trials for all areas, score each |
| `optimize_to_spectrolam(model, cfg)` | Cell 2.a | ~25 | Grid sweep over plasticity/gains/noise, return best control |

**Issues:**
- Optimization is tightly coupled to analysis
- Target profiles are fixed in code (no community edit path)
- Similarity metric is black-box; hard to understand penalization logic
- No separation of "analysis" (computing spectral properties) from "optimization objective"

**Proposed Target:** Split into two modules:
- `src/jbiophysic/analysis/spectrolaminar.py` (spectrolaminar_from_trials, target_profiles, spectrolaminar_similarity, summarize_similarity)
- `src/jbiophysic/optim/spectrolaminar_objectives.py` (optimize_to_spectrolam)

### A.6 Circuit Visualization (1 function, ~15 lines)

| Function | Location | Lines | Purpose |
|----------|----------|-------|---------|
| `visualize_circuit(model, cfg)` | Cell 2.a | ~15 | 3D scatter of neuron positions by cell type, fallback to matplotlib |

**Issues:**
- Hard to read 3D scatter in matplotlib
- Tries to import `visualize_network_3d` from jbiophysic.viz.network3d (doesn't exist yet)
- No Plotly option
- No edge rendering (could show connectivity)

**Proposed Target:** `src/jbiophysic/viz/network3d.py` (new module)

### A.7 Spectrolaminar Visualization (1 function, ~30 lines)

| Function | Location | Lines | Purpose |
|----------|----------|-------|---------|
| `plot_spectrolaminar_suite(specs, stage, cfg)` | Cell 2.a | ~30 | 3-panel plot: cell distribution, power heatmap, alpha/gamma profiles |

**Issues:**
- Uses matplotlib (static PNG only)
- Hardcodes cell type colors and markers
- No interactive hover labels
- Cannot easily edit target profile visualization

**Proposed Target:** `src/jbiophysic/viz/spectrolaminar.py` (convert to Plotly)

### A.8 Activity Suit Visualization (1 function, ~20 lines)

| Function | Location | Lines | Purpose |
|----------|----------|-------|---------|
| `activity_suit(trials, stage, cfg)` | Cell 2.a | ~20 | 3×n_areas subplot: spike raster, LFP, CSD heatmap |

**Issues:**
- Matplotlib static output
- Only first trial shown
- No interactive exploration of other trials

**Proposed Target:** `src/jbiophysic/viz/activity.py` (convert to Plotly)

### A.9 Artifact Writing (1 function, ~15 lines)

| Function | Location | Lines | Purpose |
|----------|----------|-------|---------|
| `save_ifne(model, opt_log, best_control, post_scores, cfg)` | Cell 2.a | ~15 | Pickle model dict, write manifest JSON |

**Issues:**
- Manifest is minimal (no FieldSolution solver status)
- No version control on model schema
- Pickled dict is fragile (schema changes break old artifacts)

**Proposed Target:** Enhanced in `src/jbiophysic/models/tfne_spectrolaminar.py`

### A.10 Notebook Cells (Setup, Config Display, Model Build, Sim, Optimize, Post-op, Plots, Save)

**Issues:**
- Heavy mixing of logic and narrative
- Cell dependencies are implicit
- Hidden global state (config, model)
- Hard to reuse individual pieces

**Proposed Target:** `tutorials/05_tfne_izhikevich_spectrolaminar_motif.ipynb` (thin wrapper) + `examples/tfne_izhikevich_spectrolaminar_motif.py` (standalone script)

---

## Part B: Proposed Module Architecture

### Module 1: Configuration (`src/jbiophysic/configs/spectrolaminar.py`)

**Dataclass:**
```python
@dataclass(frozen=True)
class SpectrolaminarMotifConfig:
    # Anchors
    cx_m: float = 1.0e-3
    cy_m: float = 1.0e-3
    cz_m: float = 1.0e-3
    dt_ms: float = 0.1
    n_neuron_per_column: int = 200
    
    # Simulation degrees of freedom (optimizable)
    base_plasticity: float = 0.10
    base_noise_scale: float = 1.00
    base_local_exc_gain: float = 1.00
    base_local_inh_gain: float = 1.00
    base_feedforward_gain: float = 1.00
    base_feedback_gain: float = 1.00
    
    # Anatomy (non-optimizable)
    area_order: tuple[str, ...] = ("V1", "V4", "PFC")
    cell_types: tuple[str, ...] = ("E", "PV", "SST", "VIP")
    layer_fractions: dict[str, tuple[float, float]] = ...  # L1-L6 bounds
    layer_count_frac: dict[str, float] = ...                # Fraction per layer
    fracs_layer: dict[str, dict[str, float]] = ...          # Cell type fraction per layer
    
    # Connectivity (non-optimizable)
    p_local_e: float = 0.18
    p_local_i: float = 0.30
    p_feedforward: float = 0.060
    p_feedback: float = 0.055
    w_e_range: tuple[float, float] = (0.012, 0.055)
    w_i_range: tuple[float, float] = (-0.145, -0.055)
    # ... more connectivity params
    
    # TFNE readout (non-optimizable, except via source calibration)
    tfne_grid_h_rel: float = 0.075
    tfne_source_radius_rel: float = 0.040
    tfne_conductivity_s_m: float = 0.30
    tfne_jacobi_steps: int = 16
    source_scale_a_per_native: float = 1.0e-13
    n_contacts: int = 32
    
    # Spectrolaminar readout target (fixed, not optimized)
    target_ab: np.ndarray = ...         # Alpha/beta profile [0.05, 0.15, ...]
    target_gm: np.ndarray = ...         # Gamma profile [0.95, 1.00, ...]
    band_ranges_hz: dict[str, tuple[float, float]] = ...
    
    # Runtime
    seed: int = 20260512
    t_ms_default: float = 1000.0
    n_trials: int = 10
    opt_trials: int = 10
    opt_max_evals: int = 48
    similarity_target: float = 80.0
    
    # Truth status
    truth_mode: str = "truth_safe_unverified"
    claim_level: str = "smoke_test"
```

**Methods:**
- `from_yaml(path: str) -> SpectrolaminarMotifConfig`
- `to_dict() -> dict`
- `with_smoke_defaults() -> SpectrolaminarMotifConfig` (for CI: 2-minute runs)
- `validate() -> bool` (check ranges, consistency)

**YAML Template:**
```yaml
# configs/tfne_spectrolaminar_motif.yaml
anchors:
  cx_m: 1.0e-3
  cy_m: 1.0e-3
  cz_m: 1.0e-3
  dt_ms: 0.1
  n_neuron_per_column: 200

# ... rest of config in YAML
```

**Test:** `tests/test_spectrolaminar_config.py`
- Load from YAML
- Validate smoke mode
- Serialize round-trip

### Module 2: Model Builder (`src/jbiophysic/models/spectrolaminar_motif.py`)

**Functions:**
```python
def build_spectrolaminar_motif_model(
    config: SpectrolaminarMotifConfig
) -> SpectrolaminarMotifModel:
    """Build V1/V4/PFC 3-column network with E/PV/SST/VIP neurons."""

def build_multiarea_laminar_connectivity(
    neurons: pd.DataFrame,
    positions_m: np.ndarray,
    config: SpectrolaminarMotifConfig
) -> dict[str, np.ndarray]:
    """Return {'local_exc', 'local_inh', 'feedforward', 'feedback'} matrices."""

def simulate_izhikevich_emitters(
    model: SpectrolaminarMotifModel,
    control: dict[str, float],
    seed: int,
    config: SpectrolaminarMotifConfig
) -> IzhikevichSimulation:
    """Run Izhikevich dynamics, return spikes, voltage, source state."""

def simulate_spectrolaminar_trials(
    model: SpectrolaminarMotifModel,
    control: dict[str, float],
    n_trials: int = None,
    config: SpectrolaminarMotifConfig = None,
    seed_offset: int = 0
) -> list[IzhikevichSimulation]:
    """Batch simulate with different seeds."""
```

**Dataclasses:**
```python
@dataclass(frozen=True)
class SpectrolaminarMotifModel:
    neurons: pd.DataFrame
    positions_m: np.ndarray
    connectivity_parts: dict[str, np.ndarray]  # local_exc, local_inh, FF, FB
    tfne_basis: dict[str, TFNEReadoutBasis] | None
    truth_status: str = "truth_safe_unverified"

@dataclass(frozen=True)
class IzhikevichSimulation:
    time_ms: np.ndarray
    dt_ms: float
    spikes: np.ndarray                  # (steps, n_neurons)
    voltage_mv: np.ndarray
    source_native: np.ndarray           # Raw spike/resonance source
    control: dict[str, float]
```

**Test:** `tests/models/test_spectrolaminar_motif.py`
- smoke build
- cell counts
- connectivity density
- position ranges
- no autapses
- determinism (same seed → same spikes)

### Module 3: TFNE Readout Bridge (`src/jbiophysic/models/tfne_spectrolaminar.py`)

**Functions:**
```python
def build_tfne_readout_basis(
    model: SpectrolaminarMotifModel,
    config: SpectrolaminarMotifConfig
) -> dict[str, TFNEReadoutBasis]:
    """Precompute LFP/CSD bases from neurons in each area.
    
    **CRITICAL:** Use solution.phi_e from FieldSolution and preserve
    residual_norm, n_iterations, converged, gauge_applied, boundary_condition.
    """

def tfne_readout_trial(
    model: SpectrolaminarMotifModel,
    sim: IzhikevichSimulation,
    config: SpectrolaminarMotifConfig
) -> dict[str, TFNETrialReadout]:
    """Project source currents to LFP/CSD contacts."""
```

**Dataclasses:**
```python
@dataclass(frozen=True)
class TFNEReadoutBasis:
    lfp_basis: np.ndarray                  # (n_neurons, n_contacts)
    csd_basis: np.ndarray
    contact_depths_m: np.ndarray
    basis_conservation_max_abs: float
    solver_residual_norms: np.ndarray      # One per neuron basis solve
    solver_converged_count: int
    solver_mean_iterations: float
    gauge_applied: str
    boundary_condition: str
    source_calibration_status: str = "smoke_test"

@dataclass(frozen=True)
class TFNETrialReadout:
    spikes: np.ndarray
    voltage_mv: np.ndarray
    lfp_contacts: np.ndarray               # (steps, n_contacts)
    csd_contacts: np.ndarray
    contact_depths_m: np.ndarray
    neurons: pd.DataFrame
    basis_conservation_max_abs: float
```

**Test:** `tests/models/test_tfne_spectrolaminar_readout.py`
- smoke basis build
- phi_e extraction from FieldSolution
- residual, gauge, BC recorded
- LFP/CSD projection finite
- conservation error reasonable

### Module 4: Spectrolaminar Analysis (`src/jbiophysic/analysis/spectrolaminar.py`)

**Functions:**
```python
def spectrolaminar_from_trials(
    trials: list[dict],
    area: str,
    signal_key: str = "csd_contacts",
    config: SpectrolaminarMotifConfig = None
) -> SpectrolaminarProfile:
    """Compute FFT, interpolate to contacts, normalize."""

def spectrolaminar_similarity(
    spec: SpectrolaminarProfile,
    config: SpectrolaminarMotifConfig = None,
    target_ab: np.ndarray = None,
    target_gm: np.ndarray = None
) -> float:
    """Compute multi-term error score."""

def summarize_similarity(
    trials: list[dict],
    config: SpectrolaminarMotifConfig = None
) -> tuple[pd.DataFrame, dict[str, SpectrolaminarProfile]]:
    """Score all areas in trials."""
```

**Dataclass:**
```python
@dataclass(frozen=True)
class SpectrolaminarProfile:
    freq_hz: np.ndarray
    pos_from_l4: np.ndarray
    relative_power: np.ndarray             # (n_freqs, n_contacts)
    alpha_beta: np.ndarray                 # Normalized depth profile
    gamma: np.ndarray
    n_trials: int
    n_neurons: int
    area: str
```

**Test:** `tests/analysis/test_spectrolaminar_analysis.py`
- FFT finite
- normalization bounds
- similarity score reasonable

### Module 5: Spectrolaminar Optimization (`src/jbiophysic/optim/spectrolaminar_objectives.py`)

**Functions:**
```python
def optimize_to_spectrolaminar(
    model: SpectrolaminarMotifModel,
    config: SpectrolaminarMotifConfig
) -> tuple[dict[str, float], pd.DataFrame]:
    """Grid sweep, return best control and log."""
```

**Important Doctrine:**
Optimization sweeps plasticity, synaptic gains, and noise. It does NOT directly optimize alpha/beta or gamma outputs. Those are **readout objectives**, not generator parameters.

**Test:** `tests/optim/test_spectrolaminar_optim.py`
- smoke optimize
- best control returned
- log structure
- no NaN scores

### Module 6: Circuit 3D Visualization (`src/jbiophysic/viz/network3d.py`)

**Function:**
```python
def plot_cortical_network_3d(
    neuron_table: pd.DataFrame,
    edge_table: pd.DataFrame | None = None,
    *,
    x_col: str = "x_m",
    y_col: str = "y_m",
    z_col: str = "z_m",
    color_by: str = "cell_type",
    symbol_by: str | None = "area",
    show_edges: bool = False,
    max_edges: int = 5000,
    title: str | None = None,
    size_col: str | None = None,
    color_map: dict[str, str] | None = None,
) -> "plotly.graph_objects.Figure":
    """Plotly 3D scatter of neurons, optional edges."""
```

**Test:** `tests/viz/test_network3d.py`
- returns Figure
- contains neuron count in trace
- write HTML works
- smoke data
- no display required

### Module 7: Spectrolaminar Visualization (`src/jbiophysic/viz/spectrolaminar.py`)

**Function:**
```python
def plot_spectrolaminar_suite(
    spec: SpectrolaminarProfile,
    neuron_table: pd.DataFrame | None = None,
    *,
    target_profiles: dict[str, np.ndarray] | None = None,
    title: str | None = None,
) -> "plotly.graph_objects.Figure":
    """3-panel Plotly: cell dist, power heatmap, alpha/gamma."""
```

**Panels:**
1. Left: Laminar cell density by E/PV/SST/VIP
2. Middle: Relative power heatmap (depth × frequency)
3. Right: Alpha/beta and gamma depth profiles

**Test:** `tests/viz/test_spectrolaminar_viz.py`
- returns Figure
- write HTML works
- smoke data
- target profile overlay optional

### Module 8: Activity Suit Visualization (`src/jbiophysic/viz/activity.py`)

**Function:**
```python
def plot_raster_lfp_csd_suite(
    time_ms: np.ndarray,
    spikes: np.ndarray,                 # (steps, n_neurons)
    lfp_contacts: np.ndarray,           # (steps, n_contacts)
    csd_contacts: np.ndarray,
    contact_depths_m: np.ndarray,
    *,
    neuron_table: pd.DataFrame | None = None,
    area_name: str | None = None,
    title: str | None = None,
) -> "plotly.graph_objects.Figure":
    """Multi-panel Plotly: raster, LFP, CSD heatmap."""
```

**Test:** `tests/viz/test_activity_viz.py`
- returns Figure
- write HTML works
- smoke data
- no NaN/Inf

### Module 9: EEG/MEG Toy Visualization (`src/jbiophysic/viz/eeg_meg.py`)

**Function:**
```python
def plot_eeg_meg_toy_projection(
    time_ms: np.ndarray,
    source_traces: np.ndarray,         # (steps, n_sources)
    synthetic_leadfield: np.ndarray,   # (n_sensors, n_sources)
    sensor_table: pd.DataFrame | None = None,
    *,
    modality: Literal["EEG", "MEG", "toy"] = "toy",
    title: str | None = None,
) -> "plotly.graph_objects.Figure":
    """Toy EEG/MEG projection (NO validated head model)."""
```

**CRITICAL CAVEAT:**
```
This function produces a demonstration toy projection using a synthetic leadfield.
It is NOT a validated biophysical EEG/MEG model. No EEG/MEG physical amplitude
claims are made without a real head model, validated conductivities, and SI units.
Use only for teaching demonstrations, not as evidence of EEG/MEG effects.
claim_level: computational_demo (not smoke_test, not production)
```

**Test:** `tests/viz/test_eeg_meg_viz.py`
- returns Figure
- write HTML works
- no claim of biological validity in docstring

---

## Part C: FieldSolution Compatibility Checklist

### Critical Point: Line 374 in `build_tfne_basis`

**Current Code:**
```python
phi = jacobi_poisson_neumann_smoke(eta, grid, conductivity_s_m=cfg.TFNE_CONDUCTIVITY_S_M, steps=cfg.TFNE_JACOBI_STEPS)
J = current_density(phi, Gamma, grid)
```

**After Phase 2.1 (REQUIRED CHANGE):**
```python
solution = jacobi_poisson_neumann_smoke(eta, grid, conductivity_s_m=cfg.TFNE_CONDUCTIVITY_S_M, steps=cfg.TFNE_JACOBI_STEPS)
phi = solution.phi_e
solver_status = {
    'residual_norm': solution.residual_norm,
    'n_iterations': solution.n_iterations,
    'converged': solution.converged,
    'gauge_applied': solution.gauge_applied,
    'boundary_condition': solution.boundary_condition,
}
# Propagate solver_status into TFNEReadoutBasis
J = current_density(phi, Gamma, grid)
```

### Metadata Propagation Path

1. **Solver Output (Phase 2.1):** FieldSolution(phi_e, residual_norm, n_iterations, converged, gauge, BC, solver_name, claim_level)
2. **Basis Build:** Collect residuals/status from all neuron basis solves → TFNEReadoutBasis
3. **Trial Readout:** Include basis metadata in TFNETrialReadout
4. **Manifest:** Write solver_residual_norms, solver_converged_count, gauge, BC to output JSON

---

## Part D: Risk Review

### Runtime Cost
- Smoke config: 18 neurons/column (3 cols = 54), <2 min runtime
- Full config: 200 neurons/column (3 cols = 600), ~5–10 min for one trial
- Optimization: 10 trials × 48 evals = 480 trials, **2–3 hours on single CPU**

**Mitigation:** Mark opt routines as `@pytest.mark.slow`, add smoke defaults

### Hidden Globals
- Current notebook: cfg is global in all functions
- Risk: Hard to test, easy to accidentally change

**Mitigation:** Pass config as parameter everywhere

### Physical Amplitude Overclaim
- Resonance source is designed to match spectrolaminar target
- Risk: Could be misinterpreted as biological mechanism

**Mitigation:** Document in code and README:
```
The synaptic_resonance_source is a readout scaffolding term that helps the
network express spectrolaminar resonance patterns. It is NOT a claim that
real neurons produce alpha/gamma via these mechanisms. Optimization adjusts
gains but does not validate the biological pathway.
```

### EEG/MEG Overclaim
- Toy projection with synthetic leadfield
- Risk: Could be published as if validated

**Mitigation:**
- Function name explicitly says "toy"
- Docstring warns no validated head model
- claim_level = "computational_demo"
- No figures published without "toy" label

### Plotly Optional Dependency
- Visualization requires plotly
- Risk: Makes viz optional extra, breaks if Plotly not installed

**Mitigation:**
- Add plotly to pyproject.toml as viz extra: `pip install -e '.[viz]'`
- Notebook imports with try/except

### Stale Notebook State
- Notebook cells can be run out of order
- Risk: Produces incorrect results silently

**Mitigation:**
- Tutorial notebook cells run linearly
- Each section resets variables
- Add cell tags for execution order

### Generated Outputs in Git
- Notebooks and examples write PNG/HTML/JSON
- Risk: Accidental large commits

**Mitigation:**
- Add `outputs/` to .gitignore
- Example scripts write to `/tmp` or `./outputs/`
- Manifests only saved if explicitly requested

### Motif Target Leaking into Generator
- Spectrolaminar bands are fixed readout objectives
- Risk: Optimization could treat them as tunable parameters

**Mitigation:**
- Config separates `base_control` (plasticity, gains) from `target_ab`/`target_gm` (readout only)
- Optimization loops only sweep `base_control` fields
- Target profiles documented as "fixed readout bands"

---

## Part E: Integration Phases

### Phase 2.2 (Current): Audit & Planning
**Deliverable:** This document  
**Decision:** Are module boundaries correct?  
**Gate:** Yes → Phase 2.3; No → Revise architecture

### Phase 2.3: Config + Model Builder (No Plotly)
**Add:**
- `configs/tfne_spectrolaminar_motif.yaml`
- `src/jbiophysic/configs/spectrolaminar.py`
- `src/jbiophysic/models/spectrolaminar_motif.py`
- `tests/models/test_spectrolaminar_motif.py`

**Acceptance:**
- Smoke config builds 54-neuron model
- Cell counts match
- Positions finite
- Connectivity finite
- No autapses
- Compile + pytest pass
- No biological claims

**Review:** Can community edit config without touching code?

### Phase 2.4: TFNE Readout + FieldSolution Integration
**Add:**
- `src/jbiophysic/models/tfne_spectrolaminar.py`
- `tests/models/test_tfne_spectrolaminar_readout.py`

**Changes Required:**
- Use `solution.phi_e` from FieldSolution
- Preserve `residual_norm`, `n_iterations`, `converged`, `gauge_applied`, `boundary_condition`
- Create TFNEReadoutBasis dataclass

**Acceptance:**
- Solver residuals recorded
- Gauge/BC recorded
- LFP/CSD finite
- Smoke run complete
- FieldSolution integration verified

**Review:** Does field output carry enough metadata for interpretability?

### Phase 2.5: Plotly Visualization Suites
**Add:**
- `src/jbiophysic/viz/network3d.py`
- `src/jbiophysic/viz/spectrolaminar.py`
- `src/jbiophysic/viz/activity.py`
- `src/jbiophysic/viz/eeg_meg.py`
- `tests/viz/test_*.py`

**Acceptance:**
- Each function returns Plotly Figure
- HTML write works
- No display required in CI
- Smoke data works
- No EEG/MEG overclaim

**Review:** Do figures expose structure clearly for community?

### Phase 2.6: Tutorial + Analysis/Optimization
**Add:**
- `src/jbiophysic/analysis/spectrolaminar.py`
- `src/jbiophysic/optim/spectrolaminar_objectives.py`
- `examples/tfne_izhikevich_spectrolaminar_motif.py`
- `tutorials/05_tfne_izhikevich_spectrolaminar_motif.ipynb`
- `tests/analysis/test_*.py`, `tests/optim/test_*.py`

**Acceptance:**
- Smoke mode <2 min
- Full mode >5 min (marked slow)
- Manifest written
- HTML figures saved
- No biological claims
- truth_safe_unverified stated

**Review:** Can new user run without editing hidden globals?

---

## Part F: Function Inventory Summary

| Category | Function Count | Lines | Target Module |
|----------|---|---|---|
| Config | 2 | 100 | configs/spectrolaminar.py |
| Model Build | 8 | 100 | models/spectrolaminar_motif.py |
| Emitter Sim | 4 | 120 | models/spectrolaminar_motif.py |
| TFNE Readout | 2 | 70 | models/tfne_spectrolaminar.py |
| Analysis/Score | 5 | 80 | analysis/spectrolaminar.py |
| Optimization | 1 | 25 | optim/spectrolaminar_objectives.py |
| Circuit Viz | 1 | 15 | viz/network3d.py |
| Spectro Viz | 1 | 30 | viz/spectrolaminar.py |
| Activity Viz | 1 | 20 | viz/activity.py |
| EEG/MEG Viz | — | — | viz/eeg_meg.py (new) |
| Artifact Write | 1 | 15 | models/tfne_spectrolaminar.py (enhanced) |
| **Total** | **~26** | **~575** | **9 modules** |

---

## Part G: Key Dependencies

### External Imports Required
- numpy, scipy (existing)
- pandas (existing)
- matplotlib (existing, phase 2.5+)
- plotly (new optional viz extra)
- JAX (existing)

### Internal Imports Required
- jbiophysic.cells.izhikevich (params)
- jbiophysic.tfne (make_regular_grid, gaussian_mollifier, jacobi_poisson_neumann_smoke, current_density, divergence_neumann_zero, conservation_error, assert_finite_tree)
- jbiophysic.tfne.validation (assert_no_nan_inf, assert_passive_spd)
- jbiophysic.models.tfne_izhikevich (IzhikevichTFNEScale, izh_current_to_ampere) — new module

### New Module: `src/jbiophysic/models/tfne_izhikevich.py`
```python
@dataclass(frozen=True)
class IzhikevichTFNEScale:
    """Calibration bridge from Izhikevich native units to SI amperes."""
    a_per_native: float  # Default: 1.0e-13 A per native unit

def izh_current_to_ampere(current_native: np.ndarray, scale: IzhikevichTFNEScale) -> np.ndarray:
    """Convert Izhikevich native current to SI amperes."""
```

---

## Part H: Validation & Commit Plan

### Validation Commands
```bash
# 1. Compile check
PYTHONPATH=src python -m compileall -q src tests examples

# 2. Full pytest (marked slow excluded from CI)
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=src python -m pytest -q --tb=short -m "not slow"

# 3. Drift grep (biological-proof language)
grep -R "biological proof\|biologically proven\|validated biological\|empirically proven\|real neurons" \
  README.md docs src tests examples configs || echo "✓ No overclaims found"

# 4. Git status
git status --short --branch
```

### Phase 2.2 Commit (Audit Only)
```bash
git add docs/audit/spectrolaminar_notebook_integration_plan.md
git commit -m "docs(phase2.2): plan spectrolaminar notebook integration into modular architecture

Plan spectrolaminar notebook (V1-V4-PFC TFNE-Izhikevich) integration into jbiophysic
modular stack. No code movement yet. Integration spans 9 modules:
- Config (SpectrolaminarMotifConfig dataclass + YAML)
- Model builder (network construction + connectivity)
- TFNE readout bridge (basis precomputation, FieldSolution compatibility)
- Analysis (spectrolaminar profile + similarity scoring)
- Optimization (plasticity/gains/noise sweep)
- Visualization (4 Plotly suites: circuit 3D, spectrolaminar, activity, EEG/MEG toy)
- Tutorial + example (thin narrative + standalone script)

Critical FieldSolution compatibility points identified:
- Line 374: phi = jacobi_poisson_neumann_smoke() must become solution = ...; phi = solution.phi_e
- Residual/status metadata must propagate through TFNEReadoutBasis
- All basis solves recorded with convergence status

Risks reviewed: runtime cost, hidden globals, amplitude overclaims, EEG/MEG claims,
optional deps, stale notebook state, accidental output commits, motif target leakage.

Review checkpoint: Are proposed module boundaries correct before Phase 2.3 code movement?

truth_mode: truth_safe_unverified
claim_level: smoke_test

Co-Authored-By: Claude Haiku 4.5 <noreply@anthropic.com>
"
```

---

## Part I: Decision Gate

**Phase 2.2 Completion Checklist:**

- [ ] Audit document complete (this file)
- [ ] Function inventory accurate (26 functions → 9 modules)
- [ ] FieldSolution compatibility points identified (line 374, metadata propagation)
- [ ] Risk review covers runtime, claims, dependencies (8 risks identified)
- [ ] Integration phases clearly scoped (2.3–2.6)
- [ ] Validation commands verified (compile, pytest, drift grep)
- [ ] No code moved yet (planning only)

**Review Question:** Are the module boundaries (Config → Model → TFNE → Analysis → Optim → Viz) correct before we begin Phase 2.3 code extraction?

If **YES:** Proceed to Phase 2.3 (Config + Model Builder)  
If **NO:** Revise architecture in this document before code movement  
If **UNSURE:** Flag specific boundary concerns and iterate

---

**End of Phase 2.2 Audit Document**

[claude-haiku-4-5-20251001][/Users/hamednejat/workspace/main/jbiophysic][20260513-0800]
