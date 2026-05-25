# v0.2: Passive Membrane Exercises

**Scope:** v0.2.9 teaching exercises with answer keys.  
**Level:** Undergraduate biophysics / computational neuroscience.

---

## Exercise 1: Derive Steady-State Voltage

**Problem:**

Starting from the passive membrane equation:
$$C_m \frac{dV}{dt} = -g_L(V - E_L) + I_{inj}$$

At steady state (equilibrium), $\frac{dV}{dt} = 0$.

Derive an expression for $V_{ss}$ (steady-state voltage) in terms of $E_L$, $g_L$, and $I_{inj}$.

**Answer Key:**

At equilibrium:
$$0 = -g_L(V_{ss} - E_L) + I_{inj}$$

Rearrange:
$$g_L(V_{ss} - E_L) = I_{inj}$$
$$V_{ss} - E_L = \frac{I_{inj}}{g_L}$$
$$V_{ss} = E_L + \frac{I_{inj}}{g_L}$$

**Interpretation:** 
- Positive $I_{inj}$ (inward current) increases $V_{ss}$ above $E_L$ (depolarization).
- Larger $g_L$ (more leak conductance) reduces the voltage change per unit current.
- $V_{ss}$ is independent of $C_m$ (capacitance does not affect steady state, only transient).

---

## Exercise 2: Time to Half-Maximal Deflection

**Problem:**

The exact solution to the passive membrane equation is:
$$V(t) = V_{ss} + (V_{init} - V_{ss}) e^{-t/\tau}$$

where $\tau = \frac{C_m}{g_L}$ is the time constant.

At what time $t_{half}$ does $V(t)$ reach 50% of the way from $V_{init}$ to $V_{ss}$?

**Answer Key:**

At 50% of steady state:
$$V(t_{half}) = V_{ss} - 0.5(V_{ss} - V_{init})$$
$$V_{ss} + (V_{init} - V_{ss}) e^{-t_{half}/\tau} = V_{ss} - 0.5(V_{ss} - V_{init})$$
$$(V_{init} - V_{ss}) e^{-t_{half}/\tau} = -0.5(V_{ss} - V_{init})$$
$$(V_{init} - V_{ss}) e^{-t_{half}/\tau} = 0.5(V_{init} - V_{ss})$$
$$e^{-t_{half}/\tau} = 0.5$$
$$-\frac{t_{half}}{\tau} = \ln(0.5) = -\ln(2)$$
$$t_{half} = \tau \ln(2) \approx 0.693 \tau$$

**Numerical example:**
- If $\tau = 100$ ms, then $t_{half} \approx 69.3$ ms.

**Note on confusion prevention:** The "time constant" $\tau$ is the time to reach **63.2%** of steady state (when $e^{-1} \approx 0.368$), not 50%. Common mistake!

---

## Exercise 3: Numerical Validation Against Analytical Solution

**Problem:**

Using the jbiophysic passive membrane simulator, generate a voltage trace with:
- $C_m = 100$ pF, $g_L = 1.0$ nS, $E_L = -65$ mV
- $I_{inj} = 10$ pA
- $V_{init} = -65$ mV
- $dt = 1.0$ ms, duration = 500 ms

Compare the numerical solution at $t = \tau$ (100 ms) against the analytical formula.

**Answer Key (pseudocode):**

```python
from jbiophysic.passive_membrane import (
    passive_membrane_simulate,
    PassiveMembraneParams,
    relaxation_curve,
    tau_membrane_ms,
    steady_state_voltage,
)
import numpy as np

params = PassiveMembraneParams(C_m=100.0, g_L=1.0, E_L=-65.0)
tau = tau_membrane_ms(params.C_m, params.g_L)  # = 100 ms
V_ss = steady_state_voltage(params.E_L, params.g_L, 10.0)  # = -55 mV

V_trace = passive_membrane_simulate(
    V_init=-65.0, params=params, I_inj=10.0, dt_ms=1.0, duration_ms=500.0
)

idx_tau = int(tau / 1.0)  # index at t=tau
V_numerical = V_trace[idx_tau]

V_analytical = relaxation_curve(tau, V_init=-65.0, V_ss=V_ss, tau=tau)
# Expected: V_analytical ≈ -55 + (-65 - (-55)) * exp(-1) ≈ -55 - 10 * 0.368 ≈ -59.68 mV

error = abs(V_numerical - V_analytical)
print(f"Numerical: {V_numerical:.3f} mV")
print(f"Analytical: {V_analytical:.3f} mV")
print(f"Error: {error:.4f} mV (should be <0.1 mV)")
assert error < 0.1, "Numerical solution does not match analytical!"
```

**Expected result:**
- Error < 0.1 mV (about 1% relative error).
- This validates the Euler integrator.

---

## Exercise 4: Large Timestep Artifacts

**Problem:**

Forward-Euler integration takes a single step:
$$V_{new} = V + \frac{dt}{C_m}[-g_L(V - E_L) + I_{inj}]$$

1. What happens to the magnitude of $V_{new} - V$ as $dt$ increases?
2. Is this a failure of the simulator, or a limitation of Euler's method?
3. How would you mitigate it?

**Answer Key:**

1. **Magnitude increases linearly with $dt$.** The step size is proportional to the derivative $\frac{dV}{dt}$ times $dt$.
   - Example: $dt = 0.1$ ms → $dV \approx 0.001$ mV per step
   - Example: $dt = 100$ ms → $dV \approx 1.0$ mV per step

2. **Not a simulator failure; a method limitation.** Forward Euler is a first-order method with truncation error $O(dt^2)$. For large $dt$, the local truncation error accumulates.
   - The method is *consistent* (error → 0 as $dt → 0$) but slow.
   - The method becomes unstable for $dt > 2\tau$ (the stability limit for Euler on dissipative systems).

3. **Mitigation strategies:**
   - Use $dt \ll \tau$ (typically $dt < 0.1 \tau$).
   - Switch to implicit Euler (A-stable, unconditional stability).
   - Use RK4 (4th-order, larger stable $dt$).
   - Use adaptive stepping (increase $dt$ when derivative is small, decrease near rapid changes).

---

## Exercise 5: What Transmembrane Voltage Does NOT Establish

**Problem:**

You have a soma with voltage trace $V(t)$ computed from passive membrane simulation. You observe:
- $V$ changes by 10 mV
- $\tau \approx 100$ ms
- Morphology: soma diameter 20 μm

Does this prove there is:
1. An extracellular local field potential (LFP)?
2. A recordable EEG signal?
3. A current source that will project to distant electrodes?

Why or why not? What additional information is needed?

**Answer Key:**

**1. Does this prove extracellular LFP?**
- **No.** Transmembrane voltage $V_m$ is not the same as extracellular potential $\Phi_{ext}$.
- LFP is generated by current flowing in the extracellular medium, not by voltage across the membrane.
- You need: source location, source strength, extracellular conductivity, distance to electrode, and a forward-field solver (jaxfne).

**2. Does this prove EEG?**
- **No.** EEG is recorded at the scalp surface (far from neurons).
- A single soma cannot generate detectable scalp EEG; requires synchronized, aligned population activity.
- You need: large population of neurons, alignment, conductivity model of head, and volume conduction calculations.

**3. Is there a recordable current source?**
- **Possibly, but not automatically.** Single soma with 10 mV change implies membrane current $I_{mem} = C_m dV/dt + g_L (V - E_L)$.
- But extracellular field depends on: where the current goes (soma surface area, axon initial segment, dendrites), how that current distributes in tissue, and where the recording electrode is.

**What IS needed to claim extracellular field:**

| Metadata | Status in v0.2 | Status needed for jaxfne bridge |
|----------|----------------|----------------------------------|
| Soma voltage V(t) | ✓ Computed | ✓ Available |
| Soma location (x,y,z) | ✗ Not modeled (0,0,0 assumed) | ✓ Required |
| Soma surface area | ✓ Implicit (20 μm diameter) | ✓ Required |
| Membrane current from V | ✗ Not exported | ✓ Required: $I_{mem}(t)$ |
| Source model (point, dipole, sphere) | ✗ Not defined | ✓ Required |
| Extracellular conductivity σ | ✗ Not defined | ✓ Required (measured ~0.3 S/m) |
| Domain geometry (infinite, finite, layered) | ✗ Not defined | ✓ Required |
| Probe location (x,y,z) | ✗ Not defined | ✓ Required |
| Probe impedance | ✗ Not defined | ✓ Optional (affects signal amplitude) |

**Conclusion:** Transmembrane voltage alone tells you about the neuron's electrical state. To claim extracellular field, you must provide the full source-field-probe contract (future jaxfne v0.3+ work).

---

## Exercise 6: Design a jaxfne Manifest

**Problem:**

Imagine you want to pass passive membrane current to jaxfne's forward-field operator to compute extracellular voltage at an electrode.

Design a Python data structure (dict or NamedTuple) that declares:
1. **Membrane dynamics metadata:** What was computed?
2. **Source declaration metadata:** Where and how to interpret the current as a source?
3. **Validation metadata:** How certain are you that this is correct?

**Answer Key (example):**

```python
from typing import NamedTuple

class PassiveMembraneSourceManifest(NamedTuple):
    # 1. Membrane dynamics
    C_m_pF: float  # Soma capacitance
    g_L_nS: float  # Soma conductance
    E_L_mV: float  # Leak reversal
    V_trace_mV: list  # Voltage time series
    I_mem_pA: list  # Computed membrane current (optional)
    dt_ms: float  # Timestep
    
    # 2. Source declaration
    soma_location_um: tuple  # (x, y, z)
    soma_diameter_um: float  # 20 μm
    soma_surface_area_um2: float  # π * d²
    source_type: str  # "point_dipole", "sphere", "current_sink_source"
    current_polarity: str  # "inward_positive" or "outward_positive"
    
    # 3. Validation
    truth_status: str  # "truth_safe_unverified"
    claim_type: str  # "computational_scaffold"
    empirical_validation: bool  # False (not calibrated)
    notes: str  # Free-form limitations

# Example instantiation
manifest = PassiveMembraneSourceManifest(
    C_m_pF=100.0,
    g_L_nS=1.0,
    E_L_mV=-65.0,
    V_trace_mV=[...],
    I_mem_pA=None,  # Not computed yet
    dt_ms=1.0,
    
    soma_location_um=(0.0, 0.0, 0.0),
    soma_diameter_um=20.0,
    soma_surface_area_um2=1256.6,
    source_type="point_dipole",
    current_polarity="inward_positive",
    
    truth_status="truth_safe_unverified",
    claim_type="computational_scaffold",
    empirical_validation=False,
    notes="Single soma model; no morphology; no conductivity specified; forward-field solution pending.",
)
```

**Key insights:**
- A manifest is a **contract** between simulators.
- It documents assumptions and limitations.
- It enables automated validation (e.g., "is conductivity defined? is source location physical?").
- **jaxfne must check manifests before accepting source terms.**

---

## Summary

These exercises reinforce:
1. **Mathematical literacy:** Derive formulas, not just memorize.
2. **Numerical validation:** Compare simulators against exact solutions.
3. **Practical debugging:** Recognize method limitations (large dt).
4. **Scientific integrity:** Distinguish between what you have and what you claim.
5. **Systems thinking:** Understand contracts (metadata) between components.

**Next steps:** v0.3 (Hodgkin-Huxley kinetics), v0.4 (multi-compartment), v0.5 (jaxfne bridge).
