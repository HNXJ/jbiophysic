# jbiophysic Roadmap: v0.0–v1.0+

**Date:** 2026-05-24  
**Status:** Active development (v0.3 in progress)  
**Truth mode:** truth_safe_unverified, computational_scaffold

---

## Overview

jbiophysic is an executable biophysics atlas organized in chapters (v0.x) that teach neural modeling first, computation second. Each chapter builds on prior chapters and includes:

1. **Doctrine** (`.md`) — Mathematical formulation, claim boundaries, truth status
2. **Module** (`src/jbiophysic/<chapter>/`) — Core implementations with JAX compatibility
3. **Diagnostics** — Validation (finiteness, bounds, conservation, numerical stability)
4. **Examples** (`examples/`) — Minimal reproducible demonstrations
5. **Tutorials** (`tutorials/`) — Interactive notebook walkthrough
6. **Exercises** (`docs/`) — Learning problems with answer keys
7. **Tests** (`tests/`) — Comprehensive test suite with regression checks
8. **Gate Validation** — Full test suite must pass before advancing to next chapter

---

## Chapter Status Summary

| Chapter | Phase | Status | Commits | Tests | Docs | Example | Tutorial | Gate Pass |
|---------|-------|--------|---------|-------|------|---------|----------|-----------|
| v0.0 | Identity & Hygiene | ✓ COMPLETE | b2fa08d | smoke | CLAUDE.md | — | — | — |
| v0.1 | Units & Timesteps | ✓ COMPLETE | 2a0ddc8–a1e0c24 | 31 classes | v0_1_units_doctrine.md | v0_1_units_timestep_sweep.py | —  | ✓ 61→202 tests |
| v0.2 | Passive Membrane | ✓ COMPLETE | ea23ee0–60940fa | 219 passed, 1 skipped | v0_2_membrane_doctrine.md + bridge + exercises | v0_2_passive_membrane_minimal.py | v0_2_passive_membrane_tutorial.ipynb | ✓ 202→219 tests |
| v0.3 | Hodgkin-Huxley | 🔨 IN PROGRESS | c151f02 | 0 (targeted) | v0_3_hodgkin_huxley_doctrine.md | (pending) | (pending) | ⏸ pending tests |
| v0.4–v0.6 | Multi-compartment, Plasticity, Synaptic | 📋 PLANNED | — | — | — | — | — | — |
| v0.7–v0.9 | Laminar, Network, Field | 📋 PLANNED | — | — | — | — | — | — |
| v1.0+ | Validation, Integration, Release | 📋 PLANNED | — | — | — | — | — | — |

---

## v0.0: Atlas Identity & Hygiene (✓ COMPLETE)

**Scope:** Project structure, package hygiene, import aliases.

**Status:** ✓ Accepted commit b2fa08d

**Commits:** v0.0.0–v0.0.6

**Artifacts:**
- `.claude/CLAUDE.md` — Global and project-specific operating context
- `pyproject.toml` — Package metadata, deps, extras
- `src/jbiophysic/__init__.py` — Canonical imports
- README, LICENSE, .gitignore

**Gate:** Smoke tests; no breaking imports.

---

## v0.1: Units & Timestep Discipline (✓ COMPLETE)

**Scope:** Absolute soma units (pF, nS, pA, mV, ms); integration stability; dtype compatibility.

**Status:** ✓ Accepted commit a1e0c24  
**Test Baseline:** 61 → 202 tests passed

**Key Functions:**
- `conversion_factor` — Unit scaling
- `tau_membrane_ms` — Membrane time constant
- `passive_membrane_step` — First-order forward-Euler integration
- `integration_stability_report` — JSON-safe stability checks
- `dtype_comparison` — JAX float32/float64 validation

**Documentation:** `docs/v0_1_units_doctrine.md`

**Exercises:** None (foundational chapter)

**Example:** `examples/v0_1_units_timestep_sweep.py` — Timestep sensitivity and wrong-dt null control

**Tests:** `tests/test_units_chapter1.py` — 31 test classes covering units, conversions, dtype, stability

**Gate:** ✓ Passed 202 tests, 1 skipped.

---

## v0.2: Passive Membrane Equation (✓ COMPLETE)

**Scope:** Single-compartment passive RC membrane; voltage response to step current; steady-state; numerical solution vs. exact formula.

**Status:** ✓ Accepted commit 60940fa  
**Test Baseline:** 202 → 219 tests passed

**Key Equations:**
```
C_m dV/dt = -g_L(V - E_L) + I_inj
```

**Key Functions:**
- `PassiveMembraneParams` — Membrane parameters
- `passive_membrane_simulate` — Full trajectory integration
- `steady_state_voltage` — Analytical steady state
- `relaxation_curve` — Analytical solution
- `passive_membrane_step` — Single Euler step

**Documentation:**
- `docs/v0_2_membrane_doctrine.md` — Full biophysical doctrine (150 lines)
- `docs/v0_2_jaxfne_bridge.md` — Source-field-probe contract; LFP/EEG NOT established
- `docs/v0_2_exercises.md` — 6 teaching exercises with answer keys

**Example:** `examples/v0_2_passive_membrane_minimal.py` — Voltage trace, tau, V_ss, current decomposition

**Tutorial:** `tutorials/v0_2_passive_membrane_tutorial.ipynb` — 13-section Jupyter with math, simulation, diagnostics, sweep, manifest

**Tests:** 
- `tests/test_passive_membrane_chapter2.py` — Diagnostic tests
- `tests/test_passive_membrane_chapter2_final.py` — Bridge + notebook + exercises + 19 final tests

**Gate:** ✓ Passed 219 tests, 1 skipped.

**Truth Status:** truth_safe_unverified, computational_scaffold  
**Claim Boundary:** Voltage only; does NOT establish LFP, EEG, MEG, extracellular field without source-field-probe contract.

---

## v0.3: Hodgkin-Huxley Kinetics (🔨 IN PROGRESS)

**Scope:** Single-compartment soma with voltage-gated Na and K channels; gating variables (m, h, n); action-potential-like dynamics; numerical stability.

**Status:** 🔨 Core module committed (c151f02); targeted tests PENDING  
**Test Baseline (pre-v0.3 HH tests):** 219 passed → (HH tests pending)

**Key Equations:**
```
C_m dV/dt = I_inj - I_Na - I_K - I_L

I_Na = g_Na_bar m^3 h (V - E_Na)  [outward-positive]
I_K  = g_K_bar n^4 (V - E_K)
I_L  = g_L (V - E_L)
I_ion = I_Na + I_K + I_L
I_rhs = I_inj - I_ion
dV/dt = I_rhs / C_m

dm/dt = alpha_m(V)(1-m) - beta_m(V)m
dh/dt = alpha_h(V)(1-h) - beta_h(V)h
dn/dt = alpha_n(V)(1-n) - beta_n(V)n
```

**Key Functions (v0.3.0–v0.3.3, IMPLEMENTED):**
- `HodgkinHuxleyParams` — Parameters (C_m, g_Na_bar, g_K_bar, g_L, E_Na, E_K, E_L)
- `hh_rate_functions` — Alpha/beta rate constants with L'Hôpital singularity handling (v = -40, -55)
- `hh_steady_state_gates` — Steady-state m_inf, h_inf, n_inf
- `hh_rhs_current_at_steady_gates` — Compute I_rhs at steady-state gates + zero injection
- `hh_find_rest_voltage` — Find V where I_rhs ≈ 0 (equilibrium voltage)
- `hh_currents` — I_Na, I_K, I_L, I_ion, I_rhs (outward-positive sign convention)
- `hh_step` — Single forward-Euler step
- `hh_simulate` — Full trajectory with time, V, m, h, n, I_Na, I_K, I_L, I_ion, I_rhs, all_finite

**Diagnostics (v0.3.3, IMPLEMENTED):**
- `hh_state_check` — Validate finiteness and gate bounds [0, 1]
- `hh_spike_detection` — Count spikes (threshold crossings), peak voltage, min voltage, after-hyperpolarization
- `hh_stability_report` — Comprehensive JSON-safe stability report

**Documentation (v0.3.0–v0.3.2):**
- `docs/v0_3_hodgkin_huxley_doctrine.md` — Complete doctrine (450+ lines) with:
  - HH as passive membrane + gated channels
  - Full mathematical glossary with rate functions
  - Canonical HH squid axon parameters
  - Singularity handling (L'Hôpital's rule)
  - Claim boundary: single-compartment, action-potential-like dynamics only
  - No biological validation, no extracellular field, no LFP/CSD/EEG/MEG

**Example (v0.3.4, PENDING):**
- `examples/v0_3_hodgkin_huxley_minimal.py` — Voltage, gating variables, current decomposition, phase portrait
- Default parameters: g_Na_bar=12.0, g_K_bar=3.6, g_L=0.03 (consistent scaling), dt=0.01 ms

**Tutorial (v0.3.5–v0.3.6, PENDING):**
- `tutorials/v0_3_hodgkin_huxley_tutorial.ipynb` (planned) — Interactive HH walkthrough

**Tests (v0.3.6, PENDING):**
- `tests/test_hodgkin_huxley_chapter3_core.py` (planned) — 15+ tests including:
  - Parameter creation and JSON serialization
  - Rate function finiteness and singularity handling (alpha_m at V=-40, alpha_n at V=-55)
  - Current decomposition (I_ion, I_rhs) with sign convention validation
  - Simulation finiteness and gate bounds [0, 1]
  - Action-potential-like response to step current
  - Null controls: no Na → no spike; no K → impaired repolarization; large dt → instability
  - Regression tests (v0.1–v0.2 still pass)

**Sign Convention (v0.3 CORRECTED):**
- Outward-positive ionic currents: I_Na, I_K, I_L all computed as conductance × driving force
- I_ion = I_Na + I_K + I_L (total ionic, outward-positive)
- I_rhs = I_inj - I_ion (RHS, into-cell positive)
- dV/dt = I_rhs / C_m
- NO ambiguous I_total; explicit I_ion and I_rhs in all outputs

**Rest-State Handling (v0.3 CORRECTED):**
- E_L is NOT the full resting voltage (only leak reversal potential)
- Compute V_rest via `hh_find_rest_voltage(params)` where I_rhs ≈ 0 at steady-state gates
- Use computed V_rest and steady-state gates in rest validation tests
- Parameters chosen with consistent scaling: g_Na_bar:g_K_bar:g_L = 12:3.6:0.03 (classical 120:36:0.3 scaled by 0.1)

**Gate Requirement:** Full test suite must pass (targeted HH tests + all v0.1–v0.2 regression tests) before marking v0.3 complete and advancing to v0.4–v0.6.

**Next Steps (v0.3.4–v0.3.6):**
1. Create `examples/v0_3_hodgkin_huxley_minimal.py` with consistent default parameters
2. Create comprehensive test file with all 15+ required test classes
3. Run full validation: compile check + targeted HH tests + full suite (must be 219+ passed)
4. If all pass, commit v0.3.4–v0.3.6 and mark v0.3 COMPLETE

**Truth Status:** truth_safe_unverified, computational_scaffold  
**Claim Boundary:** Single-compartment HH soma only; teaches gating kinetics and action-potential-like responses. No biological calibration, no extracellular field, no LFP/CSD/EEG/MEG.

---

## v0.4–v0.6: Multi-Compartment, Plasticity, Synaptic (📋 PLANNED)

| Phase | Feature | Scope | Status |
|-------|---------|-------|--------|
| v0.4 | Axon Initial Segment (AIS) | Two-compartment soma+AIS; relative conductances | 📋 PLANNED |
| v0.5 | Synaptic Transmission | AMPA/NMDA/GABA kinetics; conductance-based synapses | 📋 PLANNED |
| v0.6 | Short-term Plasticity | STP kernel; resource depletion/recovery model | 📋 PLANNED |

---

## v0.7–v0.9: Laminar, Network, Field (📋 PLANNED)

| Phase | Feature | Scope | Status |
|-------|---------|-------|--------|
| v0.7 | Laminar Column | L2/3, L4, L5, L6 E/PV/SST/VIP cell types; intrinsic connectivity | 📋 PLANNED |
| v0.8 | Multi-Area Network | M1 ↔ S1 ↔ V1 hierarchy; long-range connectivity; global routing | 📋 PLANNED |
| v0.9 | TFNE Forward-Field | Source (neuron) → Field (extracellular) → Probe (electrode); CSD/LFP | 📋 PLANNED |

---

## v1.0: Validation, Integration, Release (📋 PLANNED)

**Objectives:**
1. Full regression suite (v0.0–v0.9 all pass)
2. Benchmark against reference simulators (NEURON, BRIAN)
3. Finalize jaxfne integration (optional, optional [jaxfne] extra)
4. Release stable API, deprecation policy
5. Scientific documentation and citation

---

## Architecture & Separation of Concerns

### jbiophysic vs. jaxfne

**jbiophysic:**
- **Role:** Biophysics teaching atlas
- **Teaches:** Membrane, channels, synapses, networks, laminar architecture, field principles
- **Implementation:** NumPy for basic chapters (v0.1–v0.2); JAX for advanced chapters (v0.3+)
- **Claim Boundary:** Computational scaffolds only; no biological truth claims without empirical validation

**jaxfne (Optional [jaxfne] Extra):**
- **Role:** Compact JAX-native TFNE source-to-field-probe backend
- **Integration:** Used by jbiophysic ONLY in chapters v0.9+ where source-field-probe contract is explicit
- **Access:** `import jaxfne as jtfne` (canonical alias)
- **Claim Boundary:** Applies forward-field solver; requires source/field/probe metadata manifest

**Clean Separation:**
- v0.1–v0.2: Zero jaxfne dependency
- v0.3–v0.6: Zero jaxfne dependency
- v0.7–v0.8: Optional jaxfne for visualization only
- v0.9: jaxfne as backend; full source-field-probe contract

---

## Gate Validation Rule

**At every chapter boundary (v0.x.11), before advancing to v0.(x+1).0:**

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=src python -m pytest -q
```

**Requirement:** ALL tests must pass (0 failures).

**Current Gates:**
- ✓ v0.0 → v0.1: N/A (v0.0 has no tests)
- ✓ v0.1 → v0.2: 202 tests passed
- ✓ v0.2 → v0.3: 219 tests passed
- ⏸ v0.3 → v0.4: PENDING (targeted HH tests + full suite)
- — v0.4+: Future gates (TBD)

---

## Key Dependencies & Compatibility

**Core (always present):**
- numpy, scipy, pandas, PyYAML

**JAX Stack [jax] extra (required for neural modeling, v0.3+):**
- jax 0.10.0
- jaxlib 0.10.0
- equinox (differentiable PyTrees)
- optax 0.2.8 (optimizers; optional in core, required by GSDR/AGSDR)
- diffrax (ODE solvers; optional)

**jaxfne Bridge [jaxfne] extra (optional, v0.9+):**
- jaxfne 0.2.30

**Visualization [viz] extra:**
- matplotlib, plotly, dash

**Tutorials [tutorials] extra:**
- jupyter, nbformat, nbconvert, ipykernel

**Development [dev] extra:**
- pytest, pytest-cov, ruff, black

---

## Claim Boundaries by Chapter

| Chapter | Teaches | Claims | Does NOT Claim |
|---------|---------|--------|----------------|
| v0.1 | Units, timesteps, numerical stability | Proper unit scaling; Euler stability criteria | Biological validity; single-compartment adequacy |
| v0.2 | Passive RC membrane | Steady-state + transient solutions; tau; comparison to exact | Biological calibration; extracellular field; LFP/EEG/MEG |
| v0.3 | HH gating kinetics | Voltage-gated channel kinetics; action-potential-like dynamics | Biological calibration; extracellular field; whole-neuron morphology |
| v0.4–v0.6 | Multi-compartment, synapses, plasticity | Teaching architecture for distributed computation | Morphological accuracy without empirical calibration |
| v0.7–v0.8 | Laminar networks, multi-area | Network-scale connectivity; routing logic | Population-scale calibration; EEG scalability |
| v0.9 | Forward-field source-to-probe | CSD/LFP/MEG forward mapping | Inverse solutions; source localization |
| v1.0 | Validation, integration | Benchmark reproducibility; API stability | Production use without empirical context |

---

## Truth Status Convention (All Chapters)

```
truth_mode: truth_safe_unverified, computational_scaffold
empirical_validation: false
biological_claims: none
```

Every module docstring and chapter doctrine explicitly states:
- What is taught (equations, principles)
- What is NOT claimed (biological validity, field accuracy, whole-organism relevance)
- What empirical data would be required to falsify or validate

---

## Next Immediate Actions (Priority Order)

1. **v0.3.4–v0.3.6:** Create example, test file, run validation
2. **Push to origin/dev:** Complete v0.3 commit (currently 1 commit ahead due to network issues)
3. **Validate full suite:** Ensure 219+ tests pass
4. **If pass:** Mark v0.3 COMPLETE; open v0.4–v0.6 planning
5. **If fail:** Debug, fix, re-validate before push

---

**Status Summary:**
- ✓ v0.0–v0.2: COMPLETE (59 commits, 219 tests)
- 🔨 v0.3: IN PROGRESS (core module done; tests/examples pending)
- 📋 v0.4+: PLANNED (roadmap defined)

**Next Update:** After v0.3.6 validation gate passes.
