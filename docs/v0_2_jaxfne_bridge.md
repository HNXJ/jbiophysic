# v0.2.7: Passive Membrane ↔ jaxfne Source-Field Bridge

**Status:** Bridge note clarifying passive membrane's role in forward-field modeling.  
**Date:** 2026-05-23  
**Scope:** Conceptual only; jaxfne dependency optional and guarded.

---

## Overview

The passive membrane simulator outputs **membrane current** (I_ion, I_inj). To relate this to **extracellular field measurements** (LFP, CSD, EEG, MEG), one must:

1. **Declare a source model** (spatial location, orientation, morphology)
2. **Couple source current to field PDE** (conductivity, boundary conditions, solver)
3. **Project field to recording sites** (electrode geometry, gain)
4. **Validate against empirical data** (signal amplitude, frequency, latency)

This is the **source-field-probe contract** in jaxfne (v0.2.5+ roadmap).

---

## Membrane Current → Source Term

### Passive Membrane Output
```python
from jbiophysic.passive_membrane import passive_membrane_simulate, PassiveMembraneParams

params = PassiveMembraneParams(C_m=100.0, g_L=1.0, E_L=-65.0)
V_trace = passive_membrane_simulate(
    V_init=-65.0,
    params=params,
    I_inj=10.0,  # pA
    dt_ms=1.0,
    duration_ms=1000.0,
)

# Membrane dynamics: C_m dV/dt = -g_L(V - E_L) + I_inj
# Rearrange: I_ion = -g_L(V - E_L) = C_m dV/dt - I_inj
```

**Current is not automatically a source term for field modeling.**

### Required Metadata for jaxfne
To pass membrane current to jaxfne's source-field operator:

| Metadata | Value | Purpose | Status |
|----------|-------|---------|--------|
| **Morphology** | Soma location (x,y,z), radius r | 3D position in head/tissue | v0.2: not modeled |
| **Area** | Soma surface ~1260 μm² (from 20 μm diameter) | Scale current density | v0.2: documented |
| **Sign convention** | Inward (+I_inj) depolarizes | Current polarity | v0.2: defined |
| **Source type** | Point dipole, sphere, distributed | Spatial representation | v0.2: none |
| **Conductivity** | σ (S/m) of extracellular space | Field operator medium | v0.3+: forward field |
| **Boundary** | Infinite, finite, layered | Field domain shape | v0.3+: forward field |
| **Gauge** | Current-conserving, divergence-free | PDE well-posedness | v0.3+: forward field |
| **Probe geometry** | Electrode location, impedance | Recording site | v0.3+: probe module |

---

## What Passive Membrane Output DOES NOT Establish

❌ **Extracellular voltage (LFP/EEG).** Voltage V is *transmembrane*; extracellular field requires source + medium + geometry.

❌ **Current source density (CSD).** CSD is spatial Laplacian of extracellular potential; requires 2D+ field solution.

❌ **Dipole moment.** Dipole is current × distance; requires multi-compartment morphology, not single soma.

❌ **Frequency content of theta/gamma/spindles.** Membrane dynamics alone cannot claim oscillation frequency without network/synaptic context.

❌ **Signal amplitude in μV.** Field amplitude depends on source strength, distance, conductivity, and electrode impedance.

---

## jaxfne Optional Bridge (Future v0.3+)

**Current status:** Passive membrane is a source *declaration* tool.

**Future jaxfne coupling (v0.3–v0.5):**

```python
# When v0.3/jaxfne integration is ready:
try:
    import jaxfne as jtfne
    
    # 1. Declare passive membrane soma as emitter
    soma_current = passive_membrane_simulate(...)  # I_ion(t)
    
    # 2. Define source: point dipole at location
    source = jtfne.PointDipole(
        position=(0, 0, 0),  # soma center (um)
        direction=(0, 0, 1),  # axial direction
    )
    
    # 3. Define field: resistive medium
    field = jtfne.InfiniteHomogeneousField(sigma=0.3)  # S/m
    
    # 4. Define probe: electrode
    probe = jtfne.Electrode(position=(100, 0, 0))  # 100 um away
    
    # 5. Forward project
    V_electrode = field.project(source, soma_current, probe)
    
except ImportError:
    print("jaxfne not installed; source-field projection skipped.")
```

**Caveats:**
- Single soma is a poor model for extracellular field (no morphology).
- One electrode measures field very far from source (weak signal).
- Whole-brain/laminar models require multi-compartment coupling (v0.5).
- Conductivity must be measured or inferred (not assumed).

---

## Design Principle

**Passive membrane is biophysics ground truth, not signal generation.**

- ✓ Use it to teach membrane equation, stability, relaxation, units.
- ✓ Use it as a foundation for adding conductances (v0.3–v0.4).
- ✓ Use it to document source metadata for future field models.
- ✗ Do not claim it produces LFP/EEG/MEG/CSD without explicit field coupling.
- ✗ Do not overclaim amplitude, frequency, or functional relevance without validation.

---

## Summary

**v0.2.7 claim:** Passive membrane outputs membrane current, which *could* be coupled to jaxfne's source-field operator if:
1. Morphology and location are specified.
2. Conductivity and boundary conditions are defined.
3. A forward-field solver validates the projection.
4. Empirical data (if available) is compared.

**Current v0.2 scope:** None of the above. jaxfne integration is a v0.3+ feature.

**Truth status:** truth_safe_unverified; computational_scaffold.

---

**Related:** docs/v0_2_membrane_doctrine.md (physics), v0.2.8 notebook (tutorial), v0.3+ roadmap (Hodgkin-Huxley, TFNE integration).
