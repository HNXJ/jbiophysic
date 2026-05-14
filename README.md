# jbiophysic

Experimental computational neuroscience framework for:

- Izhikevich and HH-style neuron models
- laminar E/PV/SST/VIP cortical circuits
- multi-area low/mid/high cortical hierarchy simulations
- global oddball and omission task scaffolds
- TFNE forward-field CSD/LFP modeling
- optimization and plasticity experiments

## Status and Requirements

**Scope:** Exploratory research infrastructure for computational neuroscience. Not a validated biological simulator. Optimizer success is not biological proof.

**Python:** Requires Python >=3.10. Validated on Python 3.10 / 3.11 / 3.12 (GitHub Actions CI). Current suite: 106 passed, 8 skipped with full [jax,dev] install.

**Dependencies:**
- **Core:** numpy, scipy, pandas, PyYAML (minimal — no JAX required for install)
- **JAX stack:** jax, jaxlib, equinox, optax, diffrax (required for all modeling/simulation; install via [jax] extra)
- **Tutorials:** jupyter, nbformat, nbconvert, ipykernel, matplotlib (optional, [tutorials] extra)
- **Development:** pytest, pytest-cov, ruff, black (optional, [dev] extra)

**JAX & Optax Status:**
- JAX (0.10.0): All neural/field modules require JAX. CPU-safe baseline; GPU-compatible via standard JAX device model.
- Optax (0.2.8): Required by optimizer modules (AGSDR, GSDR, GSGD, SDR). Provided via [jax] extra. Core bounds/manifest helpers work without Optax.
- pmap/pjit: Current fallback-to-vmap CPU behavior is preserved; modernization is optional future work.
- PRNG: Explicit key discipline enforced; same seed → same result.

## Install

Minimal (core only):

```bash
pip install -e .
```

Development (tests — **requires [jax] for JAX-dependent tests**):

```bash
pip install -e ".[jax,dev]"
```

JAX stack (neural modeling):

```bash
pip install -e ".[jax]"
```

Tutorials (executable notebooks):

```bash
pip install -e ".[tutorials]"
```

Full stack (everything):

```bash
pip install -e ".[jax,tutorials,dev]"
```

## Quick Validation

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=src python3 -m pytest -q
```

Expected baseline may vary by branch; current archive bundle validates with targeted TFNE/JTFNE tests and strict manifest checks. Run the full command before claiming branch acceptance.


## JTFNE spectrolaminar CLI (developmental scaffold)

The spectrolaminar workflow can be run deterministically through the CLI:

    PYTHONPATH=src python scripts/run_spectrolaminar_suite.py \
      --config configs/spectrolaminar_v1.yaml \
      --seed 0 \
      --out outputs/spectrolaminar_v1_seed0 \
      --smoke

Outputs include `manifest.json`, `metrics.csv`, `celltype_diagnostics.csv`,
`field_invariants.csv`, `operator_status.json`, and `asset_hashes.json`.
These are developmental/proxy readouts under `truth_safe_unverified`; they are
not biological proof, mechanism proof, or calibrated CSD/EEG/MEG amplitude evidence.

## Tutorials

**Portable tutorials (nbconvert-executable, no magic commands):**

- `tutorials/00_neuronal_equations_book.ipynb` — Equation families overview
- `tutorials/01_izhikevich_hh_single_neurons.ipynb` — Izhikevich and HH single neurons
- `tutorials/02_tfne_forward_fields.ipynb` — TFNE forward-field modeling
- `tutorials/03_tfne_izhikevich_hybrid.ipynb` — Izhikevich-to-TFNE hybrid network
- `tutorials/04_laminar_oddball_three_area_cortex.ipynb` — Laminar cortex scaffold

These are executable teaching artifacts and should not be treated as validated biological claims. See `tutorials/README.md` for scientific guardrails and replication constraints.

**Colab artifacts (historical reference, not portable):**

- `tutorials/source_notebooks/tfne_izhikevich_net.colab.ipynb` — Original Colab notebook with google.colab imports and shell magics (%cd, !pip). For reference only; use portable tutorials for executable work.

HTML exports of portable tutorials live in `tutorials/html/`.

## Starter Examples

### TFNE-Izhikevich laminar E/I scaffold

**Executable starter simulation** for a three-layer cortical column with Izhikevich spiking neurons and TFNE extracellular field projection.

**What it demonstrates:**
- 100-neuron cylindrical cortical tube: 75 excitatory, 15 PV interneurons, 10 SST interneurons.
- Laminar density architecture (superficial, mid, deep) with heterogeneous intrinsic properties.
- Random all-to-all synaptic connectivity; inhibitory PV/SST, excitatory E.
- Izhikevich native dynamics (current-like drive, not SI nA).
- TFNE source-field projection: spike events → calibrated source amplitudes → extracellular potential via Poisson solver on cylindrical geometry.
- Automated spike-floor calibration: every neuron forced to fire at least once.

**Network composition:**
- Superficial (300 μm): 30 E, 4 PV, 7 SST = 41 neurons
- Mid (100 μm): 5 E, 5 PV, 0 SST = 10 neurons
- Deep (600 μm): 40 E, 6 PV, 3 SST = 49 neurons

**Geometry:**
- Cylindrical tube: 0.1 mm radius, 1.0 mm depth.
- TFNE grid: 25 μm resolution.
- Source smoothing radius: 20 μm.

**Run simulation:**
```bash
PYTHONPATH=src python examples/tfne_izhikevich_laminar_ei100.py --out outputs/tfne_izhikevich_laminar_ei100
```

**Generate figures** (requires matplotlib):
```bash
PYTHONPATH=src python examples/plot_tfne_izhikevich_laminar_ei100.py --out outputs/tfne_izhikevich_laminar_ei100
```

Produces:
- `figures/raster.png` — Spike raster of sample neurons
- `figures/population_rates.png` — Population firing rates by layer and cell type
- `figures/field_snapshots.png` — Extracellular potential snapshots over time

**Outputs** (in `outputs/tfne_izhikevich_laminar_ei100/`):
- `summary.json` — metadata, counts, spike floor status, layer-wise statistics
- `neuron_table.csv` — neuron anatomy and spike counts
- `spikes_and_voltage.npz` — spike raster (10000 steps × 100 neurons) and voltage traces
- `weights_post_pre.npz` — synaptic connectivity (100 × 100)
- `population_rates_1ms.csv` — firing rates by layer and cell type, 1 ms bins
- `tfne_grid.npz` — cylindrical grid geometry and active voxel mask
- `tfne_field_snapshots.npz` — extracellular potential and source density snapshots at 10 ms intervals
- `figures/` — publication-quality PNG plots (if figures script run)

**Scientific status:**
- **Exploratory scaffold only.** Not a validated biological simulator; does not claim biological truth.
- **Truth-safe unverified.** Izhikevich input is native/current-like, not SI nanoamperes. Explicit calibration constants (30/45/25 pA per spike for E/PV/SST) are toy proxies, not biophysically grounded.
- **TFNE solver is simplified:** Jacobi iteration on a regular Cartesian grid approximating cylindrical Poisson with Neumann boundary. Not a full TFNE library solver.
- **Intended use:** First scaffold toward laminar field/spectral analyses and network dynamics visualization. Not Figure-8 replication or parameter-search ground truth.

**Caveats:**
- Autapses are removed; no gap junctions.
- Synaptic time constants are fixed per cell type (E: 5 ms, PV: 8 ms, SST: 25 ms).
- Conductivity is homogeneous (0.3 S/m); no myelinated axons or inhomogeneous tissue.
- Field snapshots are sparse (10 ms stride); for finer spectral analysis, reduce `--field-stride-ms`.
