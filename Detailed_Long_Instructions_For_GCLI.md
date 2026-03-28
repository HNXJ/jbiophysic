# jbiophysics — Detailed CLI Agent Instructions

> **Project**: `jbiophysics` — A JAX/Jaxley-based biophysical neural network modeling, optimization, and visualization toolkit.
> **Environment**: Python 3.11 virtualenv at `.venv_311/bin/python`. macOS ARM64 (Apple Silicon).
> **Root Path**: `/Users/hamednejat/workspace/Computational/jbiophysics`

---

## Table of Contents

1. [Quick Start & Environment](#1-quick-start--environment)
2. [Core Mechanisms — Channel & Synapse Primitives](#2-core-mechanisms--channel--synapse-primitives)
3. [Cell Builders — Neuron Construction](#3-cell-builders--neuron-construction)
4. [Network Assembly — Column & Multi-Area Builders](#4-network-assembly--column--multi-area-builders)
5. [Fluent Builder API — NetBuilder](#5-fluent-builder-api--netbuilder)
6. [Stimulation & Context Design — Paradigm Task Flow](#6-stimulation--context-design--paradigm-task-flow)
7. [Simulation Execution — Integration & Recording](#7-simulation-execution--integration--recording)
8. [Signal Analysis — LFP, Spikes, PSD, TFR](#8-signal-analysis--lfp-spikes-psd-tfr)
9. [Optimization — SDR, GSDR, AGSDR & OptimizerFacade](#9-optimization--sdr-gsdr-agsdr--optimizerfacade)
10. [Visualization — Raster, LFP, TFR, Dashboards](#10-visualization--raster-lfp-tfr-dashboards)
11. [Export & Reporting — ResultsReport](#11-export--reporting--resultsreport)
12. [FastAPI Backend — REST Endpoints](#12-fastapi-backend--rest-endpoints)
13. [Omission Paradigm — Full Trial Pipeline](#13-omission-paradigm--full-trial-pipeline)
14. [Known Gotchas & Environment Notes](#14-known-gotchas--environment-notes)

---

## 1. Quick Start & Environment

### Installation & Activation

```bash
cd /Users/hamednejat/workspace/Computational/jbiophysics
source .venv_311/bin/activate
# OR: .venv_311/bin/python <script.py>
```

### Minimum Viable Simulation

```python
import jaxley as jx
import numpy as np
from jbiophysics.compose import NetBuilder

net = (NetBuilder(seed=42)
    .add_population("E", n=80, cell_type="pyramidal")
    .add_population("I", n=20, cell_type="pv")
    .connect("E", "all", synapse="AMPA", p=0.1)
    .connect("I", "all", synapse="GABAa", p=0.4)
    .build())

net.delete_recordings()
net.cell("all").branch(0).loc(0.0).record("v")
traces = jx.integrate(net, delta_t=0.025, t_max=1000.0)
```

### Critical Environment Variables

```bash
export MPLCONFIGDIR=/tmp/matplotlib_cache  # Bypass font cache permission errors
export XLA_FLAGS="--xla_force_host_platform_device_count=1"  # Single-device JAX
```

---

## 2. Core Mechanisms — Channel & Synapse Primitives

**File**: `jbiophysics/core/mechanisms/models.py`

These are the atomic biophysical components inserted into Jaxley cells. Every network in jbiophysics is built from combinations of these.

### `SafeHH(name="HH")` — Guarded Hodgkin-Huxley Channel

Subclasses `jaxley.channels.HH` with voltage clamping `[-100, 100] mV`, NaN guards, and gating variable clipping `[0, 1]`. **CRITICAL**: Always instantiate with `name="HH"` to ensure parameter prefixing matches downstream `cell.set("HH_gNa", ...)` calls. Using the default name will cause `KeyError`.

```python
from jbiophysics.core.mechanisms.models import SafeHH
cell.insert(SafeHH(name="HH"))
cell.set("HH_gNa", 120.0)  # Only works if name="HH"
cell.set("HH_gK", 36.0)
cell.set("HH_gLeak", 0.3)
```

**Methods**:
- `update_states(states, dt, v, params)` → clips `v` to `[-100, 100]`, applies NaN guard to gating variables `m`, `h`, `n`.
- `compute_current(states, v, params)` → clips `v` before HH current calculation.

### `Inoise(name, initial_amp_noise, initial_tau, initial_mean, initial_seed)` — Ornstein-Uhlenbeck Noise Channel

Stochastic current injection modeled as an OU process. Each neuron gets an independent noise realization via a unique seed. This is the primary source of spontaneous activity in the network.

**Parameters**:
- `amp_noise` (float): Noise amplitude (σ). Default `0.01`. Typical range: `[0.01, 0.3]`.
- `tau` (float): Time constant of OU process in ms. Default `20.0`. Higher = smoother noise.
- `mean` (float): Mean current (μ). Default `0.0`.
- `seed` (float): PRNGKey seed for JAX random generation.

```python
cell.insert(Inoise(initial_amp_noise=0.05, initial_tau=20.0, initial_mean=0.0))
```

**Methods**:
- `update_states(states, dt, v, params)` → OU drift + diffusion step with `jax.random.fold_in` for reproducible stochastic evolution.
- `compute_current(states, v, params)` → returns `-states["n"]`.
- `init_state(states, v, params, delta_t)` → initializes at `mean`.

### `GradedAMPA(g=2.5, tauD_AMPA=5.0)` — Excitatory AMPA Synapse

Graded, voltage-gated AMPA synapse with sigmoidal activation. Used for all excitatory connections (E→E, E→I, inter-areal feedforward).

**Parameters**:
- `gAMPA` (float): Peak conductance. Default `2.5` nS.
- `EAMPA` (float): Reversal potential. Fixed at `0.0` mV.
- `tauDAMPA` (float): Decay time constant. Default `5.0` ms.
- `tauRAMPA` (float): Rise time constant. Default `0.2` ms.
- `slopeAMPA`, `V_thAMPA`: Sigmoid activation shape parameters.

**Methods**:
- `update_states(states, dt, pre_v, post_v, params)` → Sigmoid activation of presynaptic voltage, first-order kinetics with NaN guard and `[0,1]` clamp.
- `compute_current(states, pre_v, post_v, params)` → `g * s * (V_post - E)`.

### `GradedGABAa(g=5.0, tauD_GABAa=5.0)` — Fast Inhibitory GABAa Synapse

Used for PV→ and SST→ inhibitory connections. Identical kinetic structure to AMPA but with inhibitory reversal `EGABAa = -80 mV`.

### `GradedGABAb(g=1.0, tauD_GABAb=200.0)` — Slow Inhibitory GABAb Synapse

Slow metabotropic inhibition. Reversal at `-95 mV`. Very long decay (`200 ms`). Used for persistent inhibitory tone modeling.

### `GradedNMDA(g=1.0, tauD_NMDA=100.0)` — Voltage-Gated NMDA Synapse

Voltage-dependent Mg²⁺ block: `m_block = 1 / (1 + 0.28 * exp(-0.062 * V_post))`. Used for inter-areal feedback connections (HO→V1) to model predictive coding.

**Mg Block**: At resting potential (~-65 mV), the channel is mostly blocked. Depolarization relieves the block, enabling coincidence detection.

---

## 3. Cell Builders — Neuron Construction

**File**: `jbiophysics/core/mechanisms/models.py` (lines 178–209)

Factory functions that return configured `jx.Cell` objects with `SafeHH(name="HH")` pre-inserted.

### `build_pyramidal_cell() → jx.Cell`

Two-compartment cell (soma + dendrite). `radius=1.0`, `length=100.0`. Standard regular-spiking parameterization.

### `build_pv_cell() → jx.Cell`

Single-compartment fast-spiking interneuron. `radius=1.0`, `length=10.0`. Used for perisomatic inhibition.

### `build_sst_cell() → jx.Cell`

Single-compartment low-threshold interneuron. `radius=1.0`, `length=10.0`. Used for dendritic inhibition / disinhibition circuits.

### `build_vip_cell() → jx.Cell`

Single-compartment VIP interneuron. `radius=0.5`, `length=10.0`. Smallest cell type. SST-targeting disinhibitory motif.

**Cell Type Parameter Setters** (in `omission_v1_column.py`):

| Function | gNa | gK | gLeak | Role |
|---|---|---|---|---|
| `_set_pyr_params(cell)` | 120.0 | 36.0 | 0.3 | Regular spiking pyramidal |
| `_set_pv_params(cell)` | 200.0 | 72.0 | 0.5 | Fast-spiking PV basket cell |
| `_set_sst_params(cell)` | 100.0 | 28.0 | 0.3 | Low-threshold SST Martinotti |
| `_set_vip_params(cell)` | 80.0 | 20.0 | 0.2 | VIP bipolar cell |

---

## 4. Network Assembly — Column & Multi-Area Builders

### V1 Laminar Column (200 neurons)

**File**: `systems/networks/omission_v1_column.py`

#### `create_v1_cells(seed, n_l23, n_l4, n_l56, n_pv, n_sst, n_vip) → (List[jx.Cell], V1PopIndices)`

Creates 200 cells organized into 6 populations following Markram et al. ratios. Returns a flat list of cells and a `V1PopIndices` dataclass containing index lists for each population.

**Default Population Sizes**: L2/3 Pyr=56, L4 Pyr=40, L5/6 Pyr=64, PV=20, SST=12, VIP=8.

**Noise Profiles**: Each population has a distinct noise amplitude ceiling:
- L2/3: `0.05`, L4: `0.08`, L5/6: `0.06`, PV: `0.04`, SST: `0.02`, VIP: `0.03`.

#### `wire_v1_column(net: jx.Network, pops: V1PopIndices)`

Applies 13 intra-columnar synaptic connectivity rules to an already-constructed `jx.Network`:
- **E→E**: L4→L2/3 AMPA (p=0.15), L2/3→L5/6 AMPA (p=0.10)
- **E→I**: L2/3→PV, L5/6→PV, L4→PV, L2/3→SST (all AMPA)
- **I→E**: PV→all_pyr GABAa (p=0.40), SST→all_pyr GABAb (p=0.25)
- **I→I**: PV→PV GABAa (p=0.30), VIP→SST GABAa (p=0.20)
- **E→I (VIP)**: L2/3→VIP AMPA (p=0.05)

#### `build_v1_column(*args, **kwargs) → (jx.Network, V1PopIndices)`

Convenience wrapper that calls `create_v1_cells` + constructs `jx.Network` + calls `wire_v1_column`. Returns assembled network and population indices.

#### `V1PopIndices` (dataclass)

```python
@dataclass
class V1PopIndices:
    l23_pyr: List[int]
    l4_pyr:  List[int]
    l56_pyr: List[int]
    pv:      List[int]
    sst:     List[int]
    vip:     List[int]

    @property
    def all_pyr(self) -> List[int]  # L2/3 + L4 + L5/6
    @property
    def all_inh(self) -> List[int]  # PV + SST + VIP
    @property
    def all(self) -> List[int]      # All 200 indices
```

### Higher-Order (HO) Column (100 neurons)

**File**: `systems/networks/omission_two_column.py`

#### `_build_ho_cells(n_l23, n_l4, n_l56, n_pv, n_sst, offset, seed) → (List[jx.Cell], HOPopIndices)`

Builds 100 cells for the higher-order / FEF column. `offset` parameter shifts all indices by the V1 neuron count (typically 200) for proper global indexing inside a merged network.

#### `HOPopIndices` (dataclass)

Same structure as `V1PopIndices` minus `vip`. Properties: `all_pyr`, `all`.

### Two-Column Omission Network (300 neurons)

#### `build_omission_network(seed=42, ...) → OmissionNetwork`

The master builder. Constructs:
1. V1 column (200 cells) via `create_v1_cells`
2. HO column (100 cells) via `_build_ho_cells` with `offset=200`
3. Merges into single `jx.Network(v1_cells + ho_cells)`
4. Applies V1 intra-columnar wiring via `wire_v1_column`
5. Applies inter-areal connections:
   - **Feedforward**: V1-L2/3 → HO-L4 AMPA (p=0.20, g=0.5)
   - **Feedback**: HO-L5/6 → V1-L2/3 AMPA+NMDA (p=0.10, g=0.3)

Returns `OmissionNetwork(net, v1_pops, ho_pops, n_v1=200, n_ho=100)`.

### Generic EIG Network

#### `build_net_eig(num_e, num_ig, num_il, seed) → jx.Network`

Legacy builder for excitatory (E) / global inhibitory (IG) / local inhibitory (IL) topology. Uses `fully_connect` for E→all and IG→all, with random sparse IL→subset.

### Generic Laminar Column

**File**: `systems/networks/laminar_column.py`

#### `build_laminar_cells(layers_config, seed) → (List[jx.Cell], meta)`

Generic laminar column from a config dict. Returns cells and metadata.

#### `build_laminar_column(layers_config, seed) → (jx.Network, meta, get_indices_fn)`

Wraps `build_laminar_cells` with network assembly and intra-columnar wiring.

### Inter-Area Connector

**File**: `systems/networks/inter_area.py`

#### `connect_cortical_areas(source_config, target_config, p_ff, p_fb, g_ff, g_fb)`

Generic FF/FB connector between two area configs. Applies Bastos/Markov laminar logic: FF targets L4, FB targets L2/3.

---

## 5. Fluent Builder API — NetBuilder

**File**: `jbiophysics/compose.py`

### `NetBuilder(seed=42)` — Composable Network Constructor

Fluent API for building networks without directly touching Jaxley internals.

#### `.add_population(name, n, cell_type, noise_amp, noise_tau, noise_mean, area) → self`

Registers a population. `cell_type` must be one of: `"pyramidal"`, `"pyr"`, `"pv"`, `"sst"`, `"vip"`. `area` parameter enables multi-area hierarchies (e.g., `area="V1"`, `area="HO"`).

#### `.connect(pre, post, synapse, p, g, area) → self`

Registers a connection rule. `synapse` must be one of: `"AMPA"`, `"GABAa"`, `"GABAb"`, `"NMDA"` (case-insensitive). `post="all"` connects to every cell. `p` is connection probability. `g` overrides default conductance.

#### `.make_trainable(params: Union[str, List[str]]) → self`

Marks synaptic parameters as independently trainable for optimization. Calls `make_synapses_independent(net, param)` internally.

#### `.build() → jx.Network`

Constructs the Jaxley network from accumulated specs. Prints confirmation with cell/connection counts.

#### `.population_offsets → Dict[str, Tuple[int, int]]`

Returns `{pop_name: (start_idx, end_idx)}` after `build()`. Essential for downstream per-population analysis.

---

## 6. Stimulation & Context Design — Paradigm Task Flow

**File**: `systems/networks/omission_v1_column.py`, `systems/networks/omission_two_column.py`

### `generate_sensory_input(t_ms, stim_times, stim_amp, pulse_width_ms) → np.ndarray`

Generates a 1D current waveform with Gaussian pulses at specified times. Default pulse width `20ms`, amplitude `2.0 nA`.

### `make_stim_schedule(t_total_ms, stim_period_ms, omission_ms, stim_amp, pulse_width_ms) → (stim_times, np.ndarray)`

Creates a periodic stimulus schedule with omission. Stimuli repeat every `stim_period_ms` but are removed after `omission_ms`.

### `make_context_inputs(config, v1_pops, ho_pops, n_v1) → np.ndarray`

**The central paradigm controller.** Returns a `(N_cells, T_steps)` current array for the entire network based on the `OmissionTrialConfig`:

| Context | `bu_on` | `td_on` | Description |
|---|---|---|---|
| 0: FF Only | `True` | `False` | Pure feedforward sensory drive to V1-L4 |
| 1: Spontaneous | `False` | `False` | No external drive, noise only |
| 2: Attended | `True` | `True` | Balanced BU + TD drive |
| 3: Omission | `False` | `True` | TD prediction without BU input |

### `OmissionTrialConfig` (dataclass)

```python
@dataclass
class OmissionTrialConfig:
    t_total_ms:     float = 5000.0
    dt_ms:          float = 0.025  # 40 kHz
    stim_period_ms: float = 500.0
    omission_ms:    float = 2500.0
    stim_amp:       float = 2.0
    td_amp:         float = 0.5    # Feedback current amplitude
    pulse_width_ms: float = 20.0
    bu_on: bool = True
    td_on: bool = False

    @property
    def n_steps(self) -> int
```

### Jaxley `stimulate()` Usage — CRITICAL API NOTE

In **Jaxley 0.13.0**, `stimulate()` expects a **1D array** `(T,)` for single-compartment views, NOT `(T, 1)`:

```python
# CORRECT:
net.cell(idx).branch(0).loc(0.0).stimulate(current_1d)  # shape (T,)

# WRONG — causes AssertionError:
net.cell(idx).branch(0).loc(0.0).stimulate(current_1d.reshape(-1, 1))  # shape (T, 1)
```

---

## 7. Simulation Execution — Integration & Recording

### Recording Setup

```python
net.delete_recordings()  # Clear previous recordings
net.cell("all").branch(0).loc(0.0).record("v")  # Record somatic voltage
```

### Integration

```python
traces = jx.integrate(net, delta_t=0.025, t_max=1000.0)
traces_np = np.array(traces)  # Convert to numpy: shape (N_cells, T_steps)
```

**Performance Notes**:
- `delta_t=0.025` ms → 40,000 steps per second of simulated time.
- 300 neurons × 40,000 steps ≈ 30–60 seconds on Apple M3 Max.
- JIT compilation on first run adds ~10–20 seconds overhead.

---

## 8. Signal Analysis — LFP, Spikes, PSD, TFR

### `extract_lfp(traces, pop_indices) → np.ndarray`

**File**: `systems/networks/omission_two_column.py` (line 345)

Virtual LFP approximation: mean membrane potential of specified pyramidal populations. Returns 1D `(T,)` signal.

```python
lfp_v1 = extract_lfp(traces_np, onet.v1_pops.l23_pyr + onet.v1_pops.l56_pyr)
```

### `detect_spikes(traces, threshold=-20.0) → Dict[int, List[int]]`

**File**: `systems/networks/omission_two_column.py` (line 357)

Threshold-crossing spike detection. Returns `{cell_idx: [spike_time_indices]}`.

### `compute_psd_numpy(signal_1d, dt, f_max=100.0) → (freqs, psd)`

**File**: `jbiophysics/viz/psd.py`

Numpy-based Power Spectral Density via FFT. Returns frequency vector and power array.

### `compute_kappa(spike_matrix, fs, bin_size_ms) → float`

**File**: `jbiophysics/core/optimizers/optimizers.py` (line 356)

Fleiss' Kappa for population synchrony measurement. Target range for physiological asynchrony: `[-0.1, 0.1]`.

### Manual Morlet CWT (in `omission_viz.py`)

`scipy.signal.morlet2` and `cwt` are **unavailable** in the current environment (scipy 1.17.1 removed them). The TFR function uses a manual implementation:

```python
def _morlet2(n, s, w=5.0):
    return np.exp(-0.5 * (n / s)**2) * np.exp(1j * w * n / s)
```

---

## 9. Optimization — SDR, GSDR, AGSDR & OptimizerFacade

**File**: `jbiophysics/core/optimizers/optimizers.py`

### `SDR(learning_rate, momentum, sigma, ...) → optax.GradientTransformation`

Stochastic Delta Rule. Gradient-sign-directed random perturbation with momentum accumulation and spatial smoothing via convolution kernels.

**Key Parameters**: `sigma=0.1` (perturbation scale), `momentum=0.9`, `change_lower_bound=-1.0`, `change_upper_bound=1.0`.

**Requires**: `key` argument passed to `update()`.

### `GSDR(inner_optimizer, ...) → optax.GradientTransformation`

Genetic Stochastic Delta Rule. Wraps an inner optimizer (typically `optax.adam`) with stochastic exploration and genetic selection:
- **Deselection**: Resets to best-known parameters when loss exceeds `deselection_threshold × best_loss`.
- **Checkpoint Reset**: After `checkpoint_n` epochs without improvement, resets to optimal.
- **Alpha Mixing**: Blends stochastic exploration (`delta`) with gradient-based updates at ratio `a`.
- **MCDP**: Multiplicative Correlated Delta Perturbation for parameter-scaled noise.

**State**: `GSDRState` dataclass tracking optimal parameters, loss, alpha, step count, and consecutive unchanged epochs.

**Key Parameters**: `a_init=0.5`, `lambda_d=1.0`, `checkpoint_n=10`, `tau_a_growth=10.0`.

### `AGSDR(inner_optimizer, ...) → optax.GradientTransformation`

Adaptive GSDR with EMA-smoothed variance-based alpha adaptation:
- `alpha = EMA(var_supervised) / (EMA(var_supervised) + EMA(var_unsupervised))`
- Clamped to `[alpha_min=0.1, alpha_max=0.9]` to prevent stochastic deadlock.
- Includes stuck-state warnings via `jax.debug.print`.

### `OptimizerFacade(net, method, lr, **kwargs)` — High-Level Optimization API

**File**: `jbiophysics/compose.py` (line 223)

Fluent interface wrapping SDR/GSDR/AGSDR with constraint specification:

```python
result = (OptimizerFacade(net, method="AGSDR", lr=1e-3)
    .set_pop_offsets(builder.population_offsets)
    .set_constraints(firing_rate=(1, 100), kappa_max=0.1)
    .set_pop_constraints("V1.E", firing_rate=(5, 30))
    .set_target(psd_profile=target_psd)
    .run(epochs=200, dt=0.025, t_max=1500.0))
```

**Loss Function Components**:
1. **Firing Rate Range**: Soft penalty for rates outside `[low, high]` Hz.
2. **Kappa Constraint**: Penalizes synchrony exceeding `kappa_max`.
3. **PSD Loss**: MSE on log-PSD against target spectral profile.
4. **Per-Population Constraints**: Independent FR and Kappa targets per area/population.

Returns a `ResultsReport` with traces, optimized parameters, and full training history.

---

## 10. Visualization — Raster, LFP, TFR, Dashboards

### Madelane Golden Dark Theme Constants

```python
GOLD   = "#CFB87C"   # Primary text, titles, axis labels
VIOLET = "#9400D3"   # HO column / accent
CYAN   = "#4FC3F7"   # V1 column / secondary
WHITE  = "#E8E8E8"   # Body text
BG     = "#0D0D0F"   # Background

LAYER_COLORS = {
    "l23": "#00FFFF", "l4": "#CFB87C", "l56": "#9400D3",
    "pv": "#FF5252", "sst": "#FF9800", "vip": "#4CAF50", "ho": "#7E57C2",
}
```

### `plot_omission_raster(traces, dt_ms, pops, threshold_mv, omission_onset_ms, title) → str`

**File**: `jbiophysics/viz/omission_viz.py`

Generates a population-colored raster plot. Returns Base64-encoded PNG. Each population is color-coded per `LAYER_COLORS`. Optional omission onset vertical line.

### `plot_lfp_traces(lfp_v1, lfp_ho, dt_ms, omission_onset_ms, title) → str`

Dual-panel V1/HO LFP plot with Gaussian smoothing (`sigma=4`) and transparent fill. Returns Base64 PNG.

### `plot_tfr(lfp, dt_ms, f_min, f_max, n_freqs, title) → str`

Time-Frequency Representation via manual Morlet CWT. Inferno colormap. Band overlays for Alpha (8-13 Hz) and Beta (13-30 Hz). dB normalization per frequency. Returns Base64 PNG.

### `generate_dashboard(report, title) → go.Figure`

**File**: `jbiophysics/viz/dashboard.py`

Plotly-based interactive dashboard with 4 panels: Voltage Traces, Spike Raster, PSD, and Mean Firing Rate histogram. Dark theme.

### `generate_laminar_dashboard(report, title) → go.Figure`

Specialized laminar analysis dashboard for spectrolaminar motif visualization.

### `plot_raster(report, threshold, title) → go.Figure`

**File**: `jbiophysics/viz/raster.py` — Plotly raster for `ResultsReport` objects.

### `plot_psd(report, window, f_max, show_bands, ...) → go.Figure`

**File**: `jbiophysics/viz/psd.py` — Plotly PSD with canonical frequency band overlays (Delta, Theta, Alpha, Beta, Gamma).

### `plot_spectrogram(report, window_ms, overlap, f_min, f_max, ...) → go.Figure`

**File**: `jbiophysics/viz/spectrogram.py` — STFT-based spectrogram with Plotly heatmap.

### `plot_traces(report, cell_indices, max_cells, title) → go.Figure`

**File**: `jbiophysics/viz/traces.py` — Interactive Plotly voltage traces for selected cells.

---

## 11. Export & Reporting — ResultsReport

**File**: `jbiophysics/export.py`

### `ResultsReport` (dataclass)

Central results container with multi-format export:

```python
@dataclass
class ResultsReport:
    traces: np.ndarray        # (N_cells, T_steps)
    params: Any = None        # Optimized parameters
    loss_history: List[float]
    dt: float = 0.1
    t_max: float = 1500.0
    metadata: Dict[str, Any]
```

**Properties**: `time_axis`, `num_cells`.

**Export Methods**:
- `.to_plotly(output_path) → go.Figure` — Interactive HTML dashboard.
- `.to_svg(output_path, panel) → str` — Static SVG export. Panels: `"raster"`, `"psd"`, `"traces"`.
- `.to_markdown(output_path, caption) → str` — Markdown summary with FR statistics.
- `.to_dict() → Dict` — JSON-serializable dictionary.
- `.save_json(path)` — Full JSON dump.
- `.from_dict(data) → ResultsReport` — Reconstruct from JSON.
- `.export(formats, output_dir, caption) → Dict[str, str]` — Batch multi-format export.

```python
report.export(formats=["plotly", "svg", "markdown"], output_dir="./results")
```

---

## 12. FastAPI Backend — REST Endpoints

**File**: `jbiophysics/api.py` — Port `7701`.

| Endpoint | Method | Description |
|---|---|---|
| `/` | GET | Interactive HTML dashboard UI |
| `/build` | POST | Generic network construction from `NetworkConfig` |
| `/simulate/{net_id}` | POST | Generic simulation of a stored network |
| `/simulate/v1` | POST | 200-neuron V1 baseline simulation |
| `/simulate/omission` | POST | Full two-column omission trial |
| `/tuning/status` | GET | Live GSDR epoch + loss during optimization |
| `/tuning/metrics` | GET | Full RSA/Kappa/SSS history per epoch |
| `/tuning/run` | POST | Start background GSDR tuning thread |
| `/visualize` | GET | Base64 PNGs from last simulation |
| `/agent/ask` | POST | Local model relay endpoint |

**Launch**:
```bash
.venv_311/bin/python jbiophysics/api.py
# Runs on http://localhost:7701
```

---

## 13. Omission Paradigm — Full Trial Pipeline

### Script: `scripts/run_omission_trial.py`

Single-context execution. Builds network, applies stimulation, integrates, analyzes, and saves visualizations to `results/`.

### Script: `scripts/generate_report_data.py`

Multi-context batch runner. Iterates through all 4 contexts (FF, Spontaneous, Attended, Omission) and saves per-context raster/LFP/TFR PNGs and a `metrics.json` to `results/data/`.

### Script: `scripts/build_html_report.py`

Assembles `results/index.html` interactive Reveal.js dashboard from generated imagery and metrics. Uses Madelane Golden Dark theme with per-context slides.

### End-to-End Pipeline

```bash
# 1. Generate data for all contexts
.venv_311/bin/python scripts/generate_report_data.py

# 2. Build HTML report
.venv_311/bin/python scripts/build_html_report.py

# 3. View results
open results/index.html
open results/REPORT.md
```

---

## 14. Known Gotchas & Environment Notes

### Jaxley 0.13.0 Specifics

1. **Mechanism Naming**: Always use `SafeHH(name="HH")`. Default naming causes `KeyError` on `cell.set("HH_gNa", ...)`.
2. **Stimulus Shape**: `stimulate()` expects `(T,)` for single-compartment views. `(T, 1)` causes `AssertionError`.
3. **Network Assembly**: When merging cells from multiple columns, create all cells first, then construct a single `jx.Network(all_cells)`, then apply wiring. Do NOT create multiple networks and try to merge them.
4. **Recording Reset**: Always call `net.delete_recordings()` before `net.cell("all").branch(0).loc(0.0).record("v")` to avoid stale state.

### Scipy 1.17.1 Specifics

- `scipy.signal.morlet2` and `scipy.signal.cwt` are **removed**. The TFR function uses a manual Morlet implementation.
- `scipy.signal.welch` is still available and used for PSD computation.

### Matplotlib Cache

The environment lacks write permissions for `~/.matplotlib` and fontconfig caches. Set `export MPLCONFIGDIR=/tmp/matplotlib_cache` before running any script that imports matplotlib to avoid 10+ second startup delays.

### Performance Tips

- Use `dt=0.025` ms for production simulations (40 kHz). Use `dt=0.1` for rapid prototyping.
- 300 neurons @ 1000ms ≈ 30s integration on M3 Max. Scales linearly with time.
- JIT warmup: First `jx.integrate` call compiles the XLA graph. Subsequent calls with same shapes are instant.
- Memory: ~2 GB for 300 neurons × 40,000 steps. Scale `t_max` accordingly.

### File Organization

```
jbiophysics/
├── jbiophysics/            # Installable package
│   ├── api.py              # FastAPI backend (port 7701)
│   ├── compose.py          # NetBuilder + OptimizerFacade
│   ├── export.py           # ResultsReport multi-format export
│   ├── core/
│   │   ├── mechanisms/
│   │   │   └── models.py   # SafeHH, Inoise, Graded{AMPA,GABAa,GABAb,NMDA}
│   │   ├── neurons/
│   │   │   └── hh_cells.py # Legacy cell definitions
│   │   └── optimizers/
│   │       └── optimizers.py # SDR, GSDR, AGSDR, compute_kappa
│   └── viz/
│       ├── dashboard.py    # Plotly interactive dashboards
│       ├── omission_viz.py # Matplotlib raster/LFP/TFR (Base64)
│       ├── psd.py          # PSD computation and plotting
│       ├── raster.py       # Plotly raster
│       ├── spectrogram.py  # STFT spectrogram
│       └── traces.py       # Plotly voltage traces
├── systems/
│   └── networks/
│       ├── omission_v1_column.py      # 200-neuron V1 column
│       ├── omission_two_column.py     # 300-neuron two-column network
│       ├── laminar_column.py          # Generic laminar builder
│       ├── inter_area.py              # FF/FB area connector
│       └── cortical_microcircuit.py   # Generic cortical microcircuit
├── scripts/
│   ├── run_omission_trial.py          # Single-context trial
│   ├── generate_report_data.py        # Multi-context batch runner
│   └── build_html_report.py           # Reveal.js report builder
└── results/
    ├── REPORT.md                      # Markdown research report
    ├── index.html                     # Interactive HTML dashboard
    └── data/                          # Per-context PNGs + metrics.json
```
