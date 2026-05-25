# v0.2: Passive Membrane Physics Doctrine

**Status:** v0.2.0–v0.2.11 doctrine, executable biophysics atlas.  
**Date:** 2026-05-23  
**truth_mode:** truth_safe_unverified, computational_scaffold  
**Scope:** Single-cell passive membrane (lumped, no spatial PDE).

---

## Overview: Passive Membrane as Electrical Circuit

The passive membrane is the foundation of neuronal biophysics. At rest and under weak stimulation, a neuron's soma behaves as a leaky capacitor in parallel with a leak resistor, driven by injected current.

**Canonical form:**

```
C_m * dV/dt = -g_L * (V - E_L) - I_inj
```

**Parameters:**

| Symbol | Units | Range | Meaning |
|--------|-------|-------|---------|
| V | mV | -100 to +50 | Membrane voltage |
| C_m | μF/cm² | 0.5–2.0 | Specific membrane capacitance |
| g_L | mS/cm² | 0.05–0.5 | Specific leak conductance |
| E_L | mV | -70 to -50 | Leak equilibrium potential |
| I_inj | pA or nA | -100 to +200 | Injected stimulus current |
| tau = C_m / g_L | ms | 10–100 | Membrane time constant |

---

## v0.2.0: Doctrine

**Identity:** Passive membrane is the pedagogical and computational ground truth for all ion-channel and synaptic effects in jbiophysic. Every neuron model (Izhikevich, HH, etc.) is either a passive membrane with added conductances or a derived reduction of biophysical kinetics.

**Design Principle:** Teach the passive membrane as a differential equation with clear physical meaning:
- **C_m**: electrical energy storage (capacitor).
- **g_L**: energy dissipation (leak conductance).
- **-I_inj**: external stimulus (boundary condition).

**Boundary Conditions:**
- V(0) = V_init (initial voltage, typically -65 mV)
- V(∞) approaches V_steady = E_L - I_inj / g_L (no limit cycle; approaches steady-state)
- Solution is exponential relaxation: V(t) = V_steady + (V_init - V_steady) * exp(-t / tau)

**Numerical Stability:**
- Forward Euler requires dt < 2 * tau (numerical stability threshold)
- For tau ~ 10–100 ms, safe dt ~ 0.01–0.1 ms
- Diagnostics check for blow-up (exponential divergence) and NaN/Inf

**Truth Status:**
- Computational scaffold; not validated against real neurons
- Teaches equation structure, numerical methods, stability
- No biological claim of accuracy; used as template for advanced models

---

## v0.2.1: Equation Glossary

**Differential equation:** C_m * dV/dt = -g_L * (V - E_L) - I_inj

**Steady-state voltage** (V_ss, when dV/dt = 0):
```
V_ss = E_L - I_inj / g_L
```

**Time constant** (tau, exponential decay rate):
```
tau = C_m / g_L
```

**Solution** (exact for constant I_inj):
```
V(t) = V_ss + (V_init - V_ss) * exp(-t / tau)
```

**Resting state** (V_rest, I_inj = 0):
```
V_rest = E_L
```

**Input resistance** (R_m = 1 / g_L):
```
R_m = 1 / g_L  [MΩ·cm²]
```

**Membrane potential deflection** (response to step current ΔI_inj):
```
ΔV_ss = -ΔI_inj / g_L  [mV per pA/μm²]
```

---

## v0.2.2: Passive Membrane Executable

**Module:** `jbiophysic.passive_membrane`

**Core functions:**

### PassiveMembraneParams (NamedTuple)

```python
PassiveMembraneParams = NamedTuple('PassiveMembraneParams', [
    ('C_m', float),           # μF/cm²
    ('g_L', float),           # mS/cm² (conductance density)
    ('E_L', float),           # mV
    ('soma_diameter', float), # μm (soma diameter; optional for whole-cell scaling)
])
```

Default soma_diameter = 20 μm (typical pyramidal neuron soma).

### passive_membrane_step(V, C_m, g_L, E_L, I_inj, dt) → V_new

Single Euler integration step:
```
V_new = V + dt / C_m * (-g_L * (V - E_L) - I_inj)
```

**Inputs:**
- V: current voltage (mV, scalar or array)
- C_m, g_L, E_L, I_inj, dt: scalars
- Supports JAX arrays (vmap/jit compatible)

**Output:**
- V_new: updated voltage (same shape as V)

### passive_membrane_simulate(V_init, params, I_inj, dt_ms, duration_ms) → V_trace

Full simulation via repeated stepping:
```python
def passive_membrane_simulate(
    V_init: float,
    params: PassiveMembraneParams,
    I_inj: float | Array,  # constant or time-varying (shape=(n_steps,))
    dt_ms: float,
    duration_ms: float,
) -> Array:
    """
    Simulate passive membrane.

    Parameters
    ----------
    V_init : float
        Initial voltage (mV)
    params : PassiveMembraneParams
        Membrane biophysical parameters
    I_inj : float or Array
        Injected current (pA); if array, shape must be (n_steps,)
    dt_ms : float
        Timestep (ms)
    duration_ms : float
        Simulation duration (ms)

    Returns
    -------
    V_trace : Array
        Voltage trace, shape (n_steps + 1,)
    """
```

**Output:** voltage trace, shape (n_steps+1,), includes initial state.

---

## v0.2.3: Diagnostics

**Module:** `jbiophysic.passive_membrane.diagnostics`

### tau_membrane_ms(C_m, g_L) → tau

Time constant in milliseconds.

```
tau = C_m / g_L  [ms]
```

For typical C_m = 1 μF/cm², g_L = 0.1 mS/cm²:
```
tau = 1 / 0.1 = 10 ms
```

### steady_state_voltage(E_L, g_L, I_inj) → V_ss

```
V_ss = E_L - I_inj / g_L  [mV]
```

For E_L = -65 mV, I_inj = 10 pA (soma area ~1000 μm²), g_L = 0.1 mS/cm²:
```
V_ss ≈ -65 - 10 / (0.1 * 1000) = -65.1 mV  [deflection ~0.1 mV]
```

### relaxation_curve(t, V_init, V_ss, tau) → V_t

Exact exponential solution:
```
V(t) = V_ss + (V_init - V_ss) * exp(-t / tau)
```

Used for theoretical comparison with numerical simulation.

### input_resistance_mohm_cm2(g_L) → R_m

Input resistance (specific):
```
R_m = 1 / g_L  [MΩ·cm²]
```

For g_L = 0.1 mS/cm²:
```
R_m = 1 / 0.1 = 10 MΩ·cm²
```

### membrane_potential_response(I_step, g_L, tau) → (V_onset, V_ss, t_half)

Characterize step current response:
- V_onset: immediate change (instantaneous if no I_Na)
- V_ss: steady-state deflection
- t_half: time to reach 50% of steady state

```python
def membrane_potential_response(
    I_step: float,  # pA (step magnitude)
    g_L: float,     # mS/cm² (conductance)
    tau: float,     # ms (time constant)
) -> dict:
    """
    Characterize passive membrane response to step current.

    Returns
    -------
    dict with keys:
        - V_ss: steady-state deflection (mV)
        - t_half: time to 50% of steady state (ms)
        - tau: time constant (ms)
    """
```

---

## v0.2.4: Voltage and Current Plots

**Module:** `jbiophysic.passive_membrane.plotting`

Visualization helpers (requires matplotlib):

### plot_voltage_trace(t, V, V_ss, label, ax=None)

Plot voltage trace with steady-state line.

### plot_iv_curve(I_range, g_L, E_L, ax=None)

Plot steady-state I-V relationship:
```
V_ss = E_L - I / g_L
```

### plot_temporal_dynamics(t, V, tau, ax=None)

Plot exponential envelope and solution together.

---

## v0.2.5: Parametric Sweep

**Module:** `jbiophysic.passive_membrane.sweep`

Sweep over C_m, g_L, I_inj and report:
- Steady-state voltage
- Time constant
- Stability (via v0.1 diagnostics)

**Example:**
```python
sweep_results = sweep_membrane_parameters(
    C_m_range=[0.5, 1.0, 2.0],  # μF/cm²
    g_L_range=[0.05, 0.1, 0.5],  # mS/cm²
    I_inj_range=[-20, 0, 20, 100],  # pA
    dt_ms=0.05,
    duration_ms=200.0,
)
```

Output: DataFrame with columns [C_m, g_L, I_inj, V_ss, tau, n_steps, is_stable, voltage_change, ...].

---

## v0.2.6: Null Test: Unstable Timestep

**Module:** `jbiophysic.passive_membrane.null_tests`

**Hypothesis:** Forward Euler diverges (blows up) when dt exceeds 2 * tau.

**Procedure:**
1. Fix C_m, g_L (hence tau = C_m / g_L)
2. Sweep dt from 0.01 * tau to 10 * tau
3. For each dt, simulate 10 steps and check for blow-up

**Expected behavior:**
- dt < 2 * tau: voltage relaxes smoothly (stable)
- dt ≈ 2 * tau: damped oscillation (borderline)
- dt > 2 * tau: exponential divergence or NaN (unstable)

**Example:**
```python
null_results = test_timestep_stability(
    C_m=1.0,
    g_L=0.1,
    I_inj=10.0,
    dt_range_ms=[0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0],
    duration_ms=50.0,
)
```

Output: DataFrame with columns [dt, tau, dt_tau_ratio, is_stable, voltage_blow_up, ...].

---

## v0.2.7: jaxfne Bridge Note

**Scope:** Passive membrane is a point-source model. To connect to jaxfne:

1. **Define Emitter:** Passive membrane soma as a current source.
2. **Define Source:** Point dipole or distributed current (soma volume) in 3D space.
3. **Define Field:** Electric field via volume conductor (e.g., infinite homogeneous medium).
4. **Define Probe:** Electrode at distance r from soma; records voltage induced by soma current.

**jaxfne contract:**
```python
import jaxfne as jtfne

# Define passive membrane as emitter
emitter_params = jtfne.IzhikevichParams(...)  # or custom PassiveMembraneEmitter
soma_current_trace = passive_membrane_simulate(...)  # I_ion shape (n_steps,)

# Define source in 3D space
source = jtfne.PointDipole(position=(0, 0, 0), direction=(0, 0, 1))

# Define conductivity field
field = jtfne.InfiniteHomogeneousField(sigma=0.3)  # S/m

# Define electrode probe
probe = jtfne.Electrode(position=(100, 0, 0))  # 100 μm away

# Forward project to electrode voltage
V_electrode = field.project(source, soma_current_trace, probe)
```

**Truth status:** Illustrative contract; full source-field validation deferred to v0.5 (laminar multi-cell).

---

## v0.2.8: Notebook

**Target:** Jupyter notebook (nbconvert-portable, no magic commands).

**Content:**
1. Review passive membrane equation and steady-state solution
2. Run minimal simulator on fixed parameters
3. Plot voltage trace and overlay exponential solution
4. Sweep C_m and g_L; show effect on tau and steady-state voltage
5. Demonstrate timestep stability threshold
6. Show wrong-timestep null (blow-up at dt > 2*tau)
7. Comment on jaxfne bridge (conceptual only)

**Outputs:** Figures (voltage traces, I-V curves, stability phase diagram).

---

## v0.2.9: Exercises

**Exercise 1:** Derive steady-state voltage for E_L = -65 mV, I_inj = 50 pA, g_L = 0.2 mS/cm².

**Exercise 2:** Calculate tau for C_m = 0.8 μF/cm², g_L = 0.1 mS/cm². What is the time to 63% of steady state? (Hint: t_63 = tau.)

**Exercise 3:** A neuron at rest (I_inj = 0) has V_init = -65 mV, E_L = -65 mV, tau = 20 ms. What is V(t=60 ms)? (Hint: no deflection; V remains at rest.)

**Exercise 4:** Simulate passive membrane with dt = 5 ms, tau = 10 ms (ratio = 0.5, stable). Why doesn't the voltage diverge even though dt is large?

**Exercise 5:** Conceptually, how would you extend the passive membrane simulator to include Hodgkin-Huxley conductances (e.g., I_Na, I_K)? (Hint: add state variables for gating.)

---

## v0.2.10: Full Test Suite (v0.2.2–v0.2.3)

**Test file:** `tests/test_passive_membrane_chapter2.py`

**Classes:**

### TestPassiveMembraneParams
- test_params_creation()
- test_params_defaults()
- test_params_json_serializable()

### TestPassiveMembraneStep
- test_step_basic()
- test_step_at_steady_state()
- test_step_with_zero_current()
- test_step_vmap_batch()

### TestPassiveMembraneSimulate
- test_simulate_basic()
- test_simulate_exponential_relaxation()
- test_simulate_steady_state_accuracy()
- test_simulate_finite_values()
- test_simulate_stability_diagnostic()

### TestMembraneProperties
- test_tau_membrane_ms()
- test_steady_state_voltage()
- test_input_resistance()
- test_relaxation_curve()

### TestMembranePhysics
- test_rest_voltage_is_leak_potential()
- test_step_current_deflection()
- test_passive_response_monotonic()

### TestNullTests
- test_timestep_stability_threshold()
- test_unstable_dt_blows_up()

**Assertions:**
- Exponential relaxation matches exact formula to within 0.1 mV over 100 ms
- Steady-state voltage accurate to 0.01 mV
- Time constant accurate to within 1%
- NaN/Inf detected immediately via v0.1 diagnostics
- dt > 2*tau triggers blow-up (checked via monotonic_blow_up_check)

---

## v0.2.11: Release Gate

**Requirement:** All tests pass (v0.2.10), full suite without warnings.

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=src python -m pytest -q --tb=short
# Expected: [N] passed, [M] skipped, 0 failed
```

**Before advancing to v0.3.0:**
- v0.2.2–v0.2.6 are implemented and tested
- All docstrings complete
- Examples run without error
- No new imports fail
- v0.1 baseline maintained (no regressions)

---

## Summary

**v0.2 Goals:**
1. Teach passive membrane as a differential equation (canonically C_m dV/dt = -g_L (V - E_L) - I_inj)
2. Implement numerically stable Euler integrator
3. Validate exponential relaxation against exact formula
4. Demonstrate timestep stability threshold (dt < 2*tau)
5. Introduce null test (unstable timestep produces divergence)
6. Provide foundation for adding ion channels (v0.3, v0.4) and spatial effects (v0.5)

**Expected deliverables:**
- 4 modules (simulator, diagnostics, null_tests, plotting)
- 2 example scripts (minimal, sweep)
- 1 notebook (temporal dynamics, I-V relationships)
- 15+ tests (all passing)
- Complete docstrings and equations

---

**Date:** 2026-05-23  
**truth_mode:** truth_safe_unverified
