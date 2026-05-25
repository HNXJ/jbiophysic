# v0.1: Units, Dimensions, and Numerical Discipline

**Goal:** Make biophysical quantities explicit before equations become complex.

---

## v0.1.0: Units Doctrine

All quantities in jbiophysic are explicitly dimensioned. Default units are:

### Electrical (SI)

| Quantity | Unit | Symbol | Notes |
|----------|------|--------|-------|
| Voltage | millivolt | mV | membrane potential relative to rest |
| Current | picoampere | pA | per neuron; sometimes nanoampere |
| Conductance | nanosiemens | nS | per synapse or channel |
| Resistance | megaohm | MΩ | per soma membrane |
| Capacitance | picofarad | pF | per soma membrane |
| Time | millisecond | ms | simulation timestep; all kinetics in ms |
| Charge | picocoulomb | pC | Q = I × t |

### Spatial

| Quantity | Unit | Symbol | Notes |
|----------|------|--------|-------|
| Length | micrometer | μm | soma, dendrite, layer |
| Area | square micrometer | μm² | soma, dendritic segment |
| Volume | cubic micrometer | μm³ | soma, voxel |
| Conductivity | siemens/meter | S/m | tissue resistivity σ |
| Current density | ampere/meter² | A/m² | source per unit area |

### Key Conversions

```
Voltage driving force:  V_drive = V_m - E_rev  (mV)
Ionic current:          I_ion = g × (V_m - E_rev)  (pA)
Membrane current:       I_m = C_m × dV/dt + I_ion + I_inj  (pA)
Charge per spike:       Q = I_avg × t_duration  (pC)
```

---

## v0.1.1: Dimensional Glossary

### Membrane Biophysics

| Symbol | Dimension | Unit | Meaning |
|--------|-----------|------|---------|
| C_m | (F/L²) | μF/cm² | membrane specific capacitance |
| R_m | (Ω·L²) | Ω·cm² | membrane specific resistance |
| τ_m = R_m × C_m | T | ms | membrane time constant |
| g_L | (S/L²) | mS/cm² | leak conductance density |
| I_inj | (A) | pA | injected current |
| V_m | (V) | mV | membrane potential |
| E_rev | (V) | mV | reversal potential |

### Hodgkin–Huxley

| Symbol | Dimension | Range | Meaning |
|--------|-----------|-------|---------|
| m, h, n | (dimensionless) | [0, 1] | gating variable activation |
| α_x(V), β_x(V) | (T⁻¹) | ms⁻¹ | voltage-dependent rate constants |
| τ_x(V) = 1/(α_x + β_x) | (T) | ms | gating time constant |
| g_Na, g_K | (S) | μS | peak sodium/potassium conductance |
| I_Na, I_K | (A) | pA | ionic current |

### Izhikevich Native Units

| Symbol | Dimension | Range | Meaning |
|--------|-----------|-------|---------|
| a | (T⁻¹) | 0.02–0.2 ms⁻¹ | recovery time scale |
| b | (S/A) | 0–0.3 pA⁻¹ | coupling strength |
| c | (V) | -90 to -50 | reset voltage |
| d | (A) | 0–400 | reset current |
| I | (A) | native units | "current-like" drive |
| v | (V) | mV or native | membrane potential (native or scaled) |
| u | (A) | native units | recovery variable |
| w = b·u | (A) | native units | effective inhibition |

⚠️ **Izhikevich issue:** Native units (a, b, c, d, I) are not SI. To use with TFNE source projection, must calibrate:
```
I_SI_pA = I_native × calibration_scale
```

See v0.1.6 (Null test) for what happens when units are wrong.

### TFNE Source & Field

| Symbol | Dimension | Unit | Meaning |
|--------|-----------|------|---------|
| I_source(r) | (A/L³) | pA/μm³ | current source density (soma-centered Gaussian) |
| σ | (S/L) | S/m | tissue conductivity |
| φ(r) | (V) | mV | extracellular potential |
| CSD = -∇²φ/σ | (A/L³) | pA/μm³ | current source density (recovered from potential) |
| LFP = φ(contact) | (V) | mV | voltage at electrode contact |

---

## v0.1.2: Unit Conversion Helpers

Create module `src/jbiophysic/units/conversions.py`:

```python
"""Unit conversion helpers for biophysical quantities."""

# Electrical conversions
def mV_to_V(x):
    """millivolt → volt"""
    return x / 1000

def V_to_mV(x):
    """volt → millivolt"""
    return x * 1000

def pA_to_nA(x):
    """picoampere → nanoampere"""
    return x / 1000

def nA_to_pA(x):
    """nanoampere → picoampere"""
    return x * 1000

def nS_to_uS(x):
    """nanosiemens → microsiemens"""
    return x / 1000

def uS_to_nS(x):
    """microsiemens → nanosiemens"""
    return x * 1000

# Spatial conversions
def um_to_mm(x):
    """micrometer → millimeter"""
    return x / 1000

def mm_to_um(x):
    """millimeter → micrometer"""
    return x * 1000

# Membrane time constants
def tau_membrane_ms(R_ohm_cm2, C_uF_cm2):
    """
    Membrane time constant.
    
    Parameters
    ----------
    R_ohm_cm2 : float
        Specific membrane resistance (Ω·cm²)
    C_uF_cm2 : float
        Specific membrane capacitance (μF/cm²)
    
    Returns
    -------
    tau_ms : float
        Time constant in milliseconds
    
    Example
    -------
    R = 10000  # Ω·cm²
    C = 1.0    # μF/cm²
    tau = tau_membrane_ms(R, C)  # → 10 ms
    """
    return R_ohm_cm2 * C_uF_cm2 / 1000  # / 1000 for ms conversion

# Conductance-area scaling
def conductance_per_soma_area(g_density_uS_cm2, soma_area_um2):
    """
    Convert conductance density to total soma conductance.
    
    Parameters
    ----------
    g_density_uS_cm2 : float
        Conductance density (μS/cm²)
    soma_area_um2 : float
        Soma surface area (μm²)
    
    Returns
    -------
    g_total_nS : float
        Total conductance (nS)
    
    Notes
    -----
    1 cm = 10,000 μm, so 1 cm² = 10^8 μm²
    """
    area_cm2 = soma_area_um2 / 1e8
    return g_density_uS_cm2 * area_cm2 * 1000  # μS → nS
```

---

## v0.1.3: Float32 vs Float64

Key numerical considerations:

### When to use float32 (JAX default)
- Neural simulations (tens of milliseconds)
- When JIT compilation is primary constraint
- Single-precision sufficient for conductances, voltages, currents

### When to use float64 (higher precision)
- Long-duration field solvers (seconds of simulation)
- Gradient-based optimization (sensitivity analysis)
- When numerical stability is marginal

### Check in notebook
- Compare voltage traces: float32 vs float64 at same seed
- Verify spike times agree (timing-sensitive? rounding issues?)
- Benchmark: is float64 actually more stable or just more expensive?

---

## v0.1.4: Stability Diagnostics

Every simulation should report:
- Min/max voltage (expected range: -90 to +30 mV)
- NaN/Inf detection (catch numerical errors early)
- Spike count (0–100 Hz typical)
- Current ranges (feasible I/O balance?)
- Time-step stability (dt too large → instability)

---

## v0.1.5: Time-Step Sweep

Test stability across dt:
- Typical: dt = 0.01–0.1 ms
- Too large: dt > 0.5 ms often unstable
- Run same seed across dt = [0.001, 0.01, 0.1, 0.5] ms
- Report: Do spike times shift? Does V remain bounded?

---

## v0.1.6: Null Test — Wrong Units

**Hypothesis:** Using Izhikevich native units as if they were SI will produce obviously wrong results.

**Test:**
```python
# Correct Izhikevich param
a_correct = 0.1  # ms⁻¹ in native units

# Wrong: treat as if pA
a_wrong = 0.1 * 1e-6  # absurdly small

# Simulate both; compare spike count, voltage range
```

Expected outcome:
- Correct: normal spiking (10–50 Hz for moderate drive)
- Wrong: no spiking, or voltage runaway, or silent

This teaches: units errors are not subtle; they break the model.

---

## v0.1.7: jaxfne Manifest Alignment

When using jaxfne backend, include in manifest:

```json
{
  "units": {
    "voltage_mV": true,
    "current_pA": true,
    "time_ms": true
  },
  "dtype": "float32",
  "dtype_note": "float32 sufficient for spike-timing; float64 for long-horizon optimization",
  "dt_ms": 0.1,
  "stability_report": {
    "v_min_max": [-80, 25],
    "has_nan": false,
    "has_inf": false,
    "spike_rate_hz": 15.3,
    "current_balance": "assessed"
  }
}
```

---

## v0.1.8–v0.1.11: Notebook, Exercises, Tests, Release

See v0.1_units_tutorial.ipynb (to be created).

---

**Next:** v0.2 (Passive Membrane Physics)
