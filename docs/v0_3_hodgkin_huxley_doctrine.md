# v0.3: Hodgkin-Huxley Kinetics Doctrine

**Status:** v0.3.0–v0.3.6 doctrine, executable biophysics atlas.  
**Date:** 2026-05-23  
**truth_mode:** truth_safe_unverified, computational_scaffold  
**Scope:** Single-compartment HH model; no morphology, no extracellular field.

---

## Overview: HH as Passive Membrane + Gated Channels

The Hodgkin-Huxley (HH) model extends the passive membrane equation by adding voltage-gated sodium and potassium conductances:

**Passive membrane (v0.2):**
$$C_m \frac{dV}{dt} = -g_L(V - E_L) + I_{inj}$$

**Hodgkin-Huxley (v0.3):**
$$C_m \frac{dV}{dt} = -g_{Na,bar} m^3 h (V - E_{Na}) - g_{K,bar} n^4 (V - E_K) - g_L(V - E_L) + I_{inj}$$

**In words:**
Membrane voltage change = injected current − sodium current − potassium current − leak current

where the sodium and potassium currents are modulated by gating variables (m, h, n) that represent the open/closed state of ion channels.

---

## v0.3.0: Doctrine

**Identity:** Hodgkin-Huxley is the simplest model that generates action potentials. It teaches:
- Voltage-gated ion channel kinetics (gates as probability variables)
- Excitability and threshold
- Regenerative feedback (positive feedback via m; negative feedback via h/n)
- Numerical stiffness in biological systems

**Design Principle:** Teach HH as an ODE system with biophysical meaning, validated against exact/reference solutions. Do NOT claim empirical accuracy or extracellular field relevance.

**Units:** Absolute soma convention (consistent with v0.2):
- V: mV
- C_m: pF (soma capacitance)
- Conductances: nS (soma total)
- Currents: pA
- Time: ms
- Gates (m, h, n): dimensionless, [0, 1]

**Claim Boundary:**
- ✓ Single-compartment action-potential-like dynamics
- ✓ Numerical stability and gating variable bounds
- ✓ Teaching scaffold for biophysical ODE systems
- ✗ No biological calibration (Hodgkin-Huxley squid, not mammalian neurons)
- ✗ No whole-neuron morphology (soma only)
- ✗ No extracellular field, LFP, CSD, EEG, MEG
- ✗ No synaptic interactions or network effects

**Truth Status:** truth_safe_unverified; computational_scaffold.

---

## v0.3.1: Mathematical Glossary

**Full HH System (4 coupled ODEs):**

$$C_m \frac{dV}{dt} = I_{inj} - I_{Na} - I_{K} - I_L$$

where:

$$I_{Na} = g_{Na,bar} \cdot m^3 \cdot h \cdot (V - E_{Na})$$
$$I_K = g_{K,bar} \cdot n^4 \cdot (V - E_K)$$
$$I_L = g_L \cdot (V - E_L)$$

**Gating Variables** (first-order kinetics):

$$\frac{dm}{dt} = \alpha_m(V)(1 - m) - \beta_m(V) \cdot m$$
$$\frac{dh}{dt} = \alpha_h(V)(1 - h) - \beta_h(V) \cdot h$$
$$\frac{dn}{dt} = \alpha_n(V)(1 - n) - \beta_n(V) \cdot n$$

**Steady-State Gates:**

$$m_\infty(V) = \frac{\alpha_m(V)}{\alpha_m(V) + \beta_m(V)}$$
$$\tau_m(V) = \frac{1}{\alpha_m(V) + \beta_m(V)}$$

(and similarly for h and n)

**Rate Functions** (Hodgkin-Huxley squid axon, 6.3°C):

$$\alpha_m(V) = \frac{0.1(V + 40)}{1 - \exp(-(V + 40)/10)}$$
$$\beta_m(V) = 4 \exp(-(V + 65)/18)$$

$$\alpha_h(V) = 0.07 \exp(-(V + 65)/20)$$
$$\beta_h(V) = \frac{1}{1 + \exp(-(V + 35)/10)}$$

$$\alpha_n(V) = \frac{0.01(V + 55)}{1 - \exp(-(V + 55)/10)}$$
$$\beta_n(V) = 0.125 \exp(-(V + 65)/80)$$

**Worded Equations:**

- **Capacitive current:** C_m dV/dt = "how fast the voltage changes"
- **Sodium current:** m^3 h (V - E_Na) = "three Na activation gates (m) and one inactivation gate (h) allow sodium in"
- **Potassium current:** n^4 (V - E_K) = "four K activation gates (n) allow potassium out"
- **Gate kinetics:** α(1 − x) − βx = "opening rate times fraction closed minus closing rate times fraction open"

**Parameters** (standard HH squid, absolute soma units):

| Symbol | Value | Unit | Meaning |
|--------|-------|------|---------|
| C_m | 100 | pF | Soma membrane capacitance |
| g_Na_bar | 12 | nS | Max Na conductance |
| g_K_bar | 3.6 | nS | Max K conductance |
| g_L | 0.3 | nS | Leak conductance |
| E_Na | +60 | mV | Na reversal potential |
| E_K | -77 | mV | K reversal potential |
| E_L | -54 | mV | Leak reversal potential |

---

## v0.3.2: Minimal Executable

**Module:** src/jbiophysic/hodgkin_huxley/

**Files:**
- `__init__.py` — exports
- `simulator.py` — HodgkinHuxleyParams, rate functions, currents, step, simulate
- `diagnostics.py` — finite checks, gate bounds, spike detection, stiffness

**Core Classes & Functions:**

### HodgkinHuxleyParams (NamedTuple)
```python
class HodgkinHuxleyParams(NamedTuple):
    C_m: float      # pF
    g_Na_bar: float # nS
    g_K_bar: float  # nS
    g_L: float      # nS
    E_Na: float     # mV
    E_K: float      # mV
    E_L: float      # mV
```

### hh_rate_functions
```python
def hh_rate_functions(V: float) -> dict:
    """Return alpha and beta rate constants at voltage V."""
    # Handle removable singularities (0/0 at special voltages)
    # Use L'Hôpital's rule or direct formula
    # Return: {α_m, β_m, α_h, β_h, α_n, β_n}
```

### hh_currents
```python
def hh_currents(V: float, m: float, h: float, n: float, 
                params: HodgkinHuxleyParams, I_inj: float) -> dict:
    """Compute I_Na, I_K, I_L given voltage and gates."""
    # Return: {I_Na, I_K, I_L, I_total}
```

### hh_step
```python
def hh_step(V: float, m: float, h: float, n: float,
            params: HodgkinHuxleyParams, I_inj: float, dt_ms: float) -> tuple:
    """Single forward-Euler step.
    Return: (V_new, m_new, h_new, n_new)
    """
```

### hh_simulate
```python
def hh_simulate(V_init: float, m_init: float, h_init: float, n_init: float,
                params: HodgkinHausleyParams, I_inj: float,
                dt_ms: float, duration_ms: float) -> dict:
    """Full simulation returning all traces."""
    # Return: {V_trace, m_trace, h_trace, n_trace, 
    #          I_Na_trace, I_K_trace, I_L_trace, time}
```

---

## v0.3.3: Diagnostics

**Finite Check:** All traces (V, m, h, n, currents) must be finite.

**Gate Bounds:** 0 ≤ m, h, n ≤ 1 (with small tolerance, e.g., ±1e-6).

**Current Decomposition:**
$$I_{total} = I_{Na} + I_K + I_L = C_m \frac{dV}{dt} - I_{inj}$$
(conservation check)

**Spike Detection:** Count threshold crossings (e.g., V > 0 mV).

**Peak Voltage:** Max V in trace.

**After-Hyperpolarization:** Min V in trace.

**Resting State:** At I_inj = 0, V should approach E_L (approx. −70 mV).

**Stiffness Warning:** If dt > 0.1 ms with g_Na_bar >> g_L, flag potential instability.

---

## v0.3.4: Visualization & Examples

**Example:** examples/v0_3_hodgkin_huxley_minimal.py

**Computed & Plotted:**
1. Voltage trace V(t) with threshold lines
2. Gating variables m(t), h(t), n(t)
3. Current decomposition: I_Na, I_K, I_L
4. Phase diagram: V vs. dV/dt (nullcline visualization optional)

**Output:** Console summary + figures (untracked).

---

## v0.3.5: Parameter Sweep

**Sweep Dimensions:**
- I_inj: 0, 5, 10, 20, 50 pA
- g_Na_bar: 6, 12, 24 nS
- g_K_bar: 1.8, 3.6, 7.2 nS
- dt: 0.01, 0.1, 1.0 ms

**Report per sweep:** Spike count, peak V, min V, all_finite, gates_valid, I_max.

---

## v0.3.6: Null Controls

**Null 1: Remove Na** (g_Na_bar = 0)
- Expected: No regenerative spike; voltage deflects passively.
- Test: Spike count = 0.

**Null 2: Remove K** (g_K_bar = 0)
- Expected: Impaired repolarization or voltage clamp instability.
- Test: Peak voltage unreasonably high (e.g., > +100 mV) or gates invalid.

**Null 3: Large dt**
- Expected: Numerical artifact (gates may escape [0,1], voltage may diverge).
- Test: dt = 10 ms → gate bounds violated or finite_check fails.

**Null 4: Wrong reversal potential scale**
- Expected: Rejection (e.g., E_Na < E_K is unphysical).
- Test: Manifest flags error; manifest.is_valid = false.

---

## Summary

**v0.3 Goals:**
1. Teach HH as an ODE system extending passive membrane.
2. Implement voltage-gated gating variables with rate functions.
3. Generate action-potential-like responses to step current.
4. Validate numerical integration and gate bounds.
5. Provide foundation for adding morphology, plasticity, network (v0.4+).

**Expected Deliverables:**
- 1 doctrine file (450 lines)
- 1 module (simulator.py + diagnostics.py, ~300 lines)
- 1 example script (~150 lines)
- 15+ tests (param creation, rates, currents, gates, spikes, nulls)
- All v0.1–v0.2 tests still pass

**Truth Status:** truth_safe_unverified; computational_scaffold.

---

**Date:** 2026-05-23  
**truth_mode:** truth_safe_unverified
