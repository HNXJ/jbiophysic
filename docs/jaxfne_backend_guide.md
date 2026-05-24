# jaxfne Backend Integration Guide

**Status:** Production-ready (Phases 1-4 complete)  
**Version:** jbiophysic 1.1.1 + jaxfne 0.2.30  
**Date:** 2026-05-23

---

## Quick Start

### Basic Usage: Switch Backend

```python
from jbiophysic import jtfne

# Build model (jaxfne EIGNetwork auto-included)
model = jtfne.construct(jtfne.JTFNEInitConfig(...))

# Simulate with default legacy backend
result_legacy = jtfne.simulate(model, cfg.sim, backend='legacy')

# Simulate with jaxfne backend
result_jaxfne = jtfne.simulate(model, cfg.sim, backend='jaxfne')

# Both produce identical output structure
assert result_legacy.trials[0]['V1']['spikes'].shape == result_jaxfne.trials[0]['V1']['spikes'].shape
```

### Advanced: Direct jaxfne Access

```python
from jbiophysic import jtfne

# Build model with jaxfne objects
model = jtfne.construct(cfg.init, include_jaxfne=True)

# Access EIGNetwork directly
eig_net = model.eig_network
edges = model.edges

# Run custom simulation
v, u, spikes = jtfne.simulate_with_jaxfne(
    eig_net,
    edges,
    n_steps=2000,
    dt_ms=0.5,
    seed=42,
)

# Project to laminar field
field = jtfne.project_to_laminar_field(spikes, eig_net.positions)
print(f"LFP shape: {field.lfp_proxy.shape}")  # (n_steps, n_contacts)
```

---

## Architecture Overview

### Three Integration Levels

| Level | API | Use Case |
|-------|-----|----------|
| **High** | `simulate(backend='jaxfne')` | Drop-in replacement for legacy |
| **Mid** | `jbiophysic_to_eig_network()` + `simulate_with_jaxfne()` | Custom pipelines |
| **Low** | `jaxfne` library directly | Advanced research |

### Data Flow

```
construct() with jaxfne
├─ Legacy path: neurons DataFrame, connectivity matrices
└─ jaxfne path: EIGNetwork + EdgeList

simulate(backend='jaxfne')
├─ simulate_receptor_exponential_izhikevich()  [neuron model]
│  └─ Returns: v(n_steps, n_neurons), u, spikes
├─ project_laminar_sources()  [field projection]
│  └─ Returns: FieldOutput with LFP/CSD/potential
└─ Organize results in legacy format
   └─ Returns: trials[area] = {spikes, voltage, lfp_contacts, ...}
```

---

## Receptor Kinetics

### Default Receptors (from jaxfne.standard_receptor_specs)

```python
from jbiophysic import jtfne

receptors = jtfne.get_receptor_info()

# AMPA (excitatory)
print(receptors['AMPA'])
# {
#   'receptor_index': 0,
#   'sign': 1,           # excitatory
#   'tau_ms': 2.0,       # fast decay
#   'reversal_mV': 0.0,
#   'source_calibration_status': 'uncalibrated_izhikevich_native_current',
#   'claim_level': 'computational_scaffold'
# }

# GABA_A (inhibitory, fast)
print(receptors['GABA_A'])
# {
#   'receptor_index': 1,
#   'sign': -1,          # inhibitory
#   'tau_ms': 5.0,       # medium decay
#   'reversal_mV': -80.0,
#   ...
# }

# NMDA (excitatory, slow)
print(receptors['NMDA'])
# {
#   'receptor_index': 2,
#   'tau_ms': 100.0,     # slow decay
#   ...
# }

# GABA_B (inhibitory, slow)
print(receptors['GABA_B'])
# {
#   'receptor_index': 3,
#   'tau_ms': 150.0,     # very slow
#   'reversal_mV': -95.0,
#   ...
# }
```

### Connectivity Mapping

| Connection | Receptor | tau_ms | Example Edge |
|------------|----------|--------|--------------|
| E→E (local) | AMPA | 2.0 | EXC local excitatory |
| E→PV (local) | AMPA | 2.0 | Feed E to fast-spiking |
| E→SST (local) | AMPA | 2.0 | Feed E to slow-spiking |
| E→VIP (local) | AMPA | 2.0 | Feed E to VIP |
| E→all (feedfwd) | AMPA | 2.0 | Cross-layer E→L4 |
| E→all (feedback) | AMPA | 2.0 | Deep→superficial |
| PV→all | GABA_A | 5.0 | Fast inhibition |
| SST→all | GABA_A | 5.0 | Medium inhibition |
| VIP→all | GABA_A | 5.0 | VIP inhibition |

---

## Connectivity Analysis

### Diagnose Network Structure

```python
from jbiophysic import jtfne

model = jtfne.construct(cfg.init)
diag = jtfne.diagnose_connectivity(model.eig_network, model.edges)

print(f"Neurons: {diag['n_neurons']}")
print(f"Synapses: {diag['n_edges']}")
print(f"Sparsity: {diag['connection_density']:.2%}")
print(f"Excitatory: {diag['excitatory_fraction']:.1%}")

# Receptor breakdown
print(f"AMPA edges: {diag['receptor_counts'].get('AMPA', 0)}")
print(f"GABA_A edges: {diag['receptor_counts'].get('GABA_A', 0)}")

# Cell type distribution
for cell_type, count in diag['cell_type_distribution'].items():
    print(f"{cell_type}: {count}")
```

---

## Workflow Comparison

### Legacy (Original)

```python
model = jtfne.construct(cfg.init)
result = jtfne.simulate(model, cfg.sim)  # backend='legacy' (default)

# Simulation path:
# 1. Custom Izhikevich stepper (NumPy-based)
# 2. Custom TFNE Poisson solver
# 3. Custom basis projection (lfp_basis, csd_basis matrices)
```

### jaxfne (New)

```python
model = jtfne.construct(cfg.init)  # includes eig_network + edges
result = jtfne.simulate(model, cfg.sim, backend='jaxfne')

# Simulation path:
# 1. jaxfne.simulate_receptor_exponential_izhikevich (JAX-compiled)
# 2. jaxfne.project_laminar_sources (Gaussian laminar proxy)
# 3. Contact-depth readout
```

### Output Compatibility

Both produce identical structure:

```python
result.trials[0]['V1'] = {
    'spikes': (n_steps, n_neurons_V1),         # boolean
    'voltage_mV': (n_steps, n_neurons_V1),     # float32
    'lfp_contacts': (n_steps, 16),             # float32
    'csd_contacts': (n_steps, 16),             # float32
    'contact_depths_m': (16,),                 # float32
    'neurons': DataFrame,
    'metadata': {'backend': 'legacy'|'jaxfne'},
}
```

---

## Performance Considerations

### JAX Compilation Benefits

- **First run:** ~2s (JIT compilation + execution)
- **Subsequent runs:** ~0.5s (cached compilation)
- **Scaling:** Linear in n_neurons and n_steps
- **Hardware:** CPU-optimized (no GPU assumption)

### Memory Efficiency

- **Legacy:** Full dense weight matrices (n² memory for n neurons)
- **jaxfne:** Sparse EdgeList (~1% memory for typical sparsity)

### Determinism

Both backends are deterministic given the same seed:

```python
# Legacy: Always same result with same seed
spikes1 = jtfne.simulate(model, cfg.sim, backend='legacy')
spikes2 = jtfne.simulate(model, cfg.sim, backend='legacy')
assert np.allclose(spikes1.trials[0]['V1']['spikes'],
                   spikes2.trials[0]['V1']['spikes'])

# jaxfne: Also deterministic
spikes1 = jtfne.simulate(model, cfg.sim, backend='jaxfne')
spikes2 = jtfne.simulate(model, cfg.sim, backend='jaxfne')
assert np.allclose(spikes1.trials[0]['V1']['spikes'],
                   spikes2.trials[0]['V1']['spikes'])
```

---

## Troubleshooting

### "ImportError: jaxfne not installed"

```bash
pip install jaxfne
pip install jax jaxlib  # dependencies
```

### "jaxfne backend requested but jaxfne not available"

This error means construct() built the model without jaxfne objects (unlikely with
default behavior). Check:

```python
model = jtfne.construct(cfg.init)
assert hasattr(model, 'eig_network')  # Should be True by default
```

Or install jaxfne:

```python
import jbiophysic.jtfne as jtfne
print(f"HAS_JAXFNE_INTEGRATION: {jtfne.HAS_JAXFNE_INTEGRATION}")
```

### "Determinism Mismatch Between Backends"

If legacy and jaxfne produce different spikes despite same seed, this is expected:
- Legacy uses NumPy RNG (Philox)
- jaxfne uses JAX RNG (ThreeFry)
- Both are deterministic within their backend; outputs differ by design

To compare across backends, look at high-level statistics (firing rates, synchrony)
rather than exact spike times.

---

## Best Practices

### 1. Always Use construct() with Default Parameters

```python
# ✓ Good: Gets jaxfne objects automatically
model = jtfne.construct(cfg.init)

# ✗ Avoid: Explicitly setting include_jaxfne=True is redundant
model = jtfne.construct(cfg.init, include_jaxfne=True)
```

### 2. Choose Backend Based on Needs

```python
# ✓ Use legacy if:
# - Compatibility with older results critical
# - Debugging (more straightforward code path)
result = jtfne.simulate(model, cfg.sim, backend='legacy')

# ✓ Use jaxfne if:
# - Performance matters (sparse networks)
# - Receptor kinetics exploration
# - Modern JAX ecosystem integration
result = jtfne.simulate(model, cfg.sim, backend='jaxfne')
```

### 3. Don't Mix Output Formats

```python
# ✗ Don't do this:
legacy_lfp = jtfne.simulate(model, cfg.sim, backend='legacy').trials[0]['V1']['lfp_contacts']
jaxfne_lfp = jtfne.simulate(model, cfg.sim, backend='jaxfne').trials[0]['V1']['lfp_contacts']
diff = np.abs(legacy_lfp - jaxfne_lfp)  # Will show differences!

# ✓ Instead, compare high-level metrics:
from jbiophysic.analysis.diagnostics import synchrony_diagnostics
sync_legacy = synchrony_diagnostics(legacy_lfp)
sync_jaxfne = synchrony_diagnostics(jaxfne_lfp)
# Now compare kappa values, not raw signals
```

### 4. Validate Connectivity Before Simulation

```python
model = jtfne.construct(cfg.init)
diag = jtfne.diagnose_connectivity(model.eig_network, model.edges)

# Sanity checks
assert diag['n_edges'] > 0, "No connectivity!"
assert diag['excitatory_fraction'] > 0.3, "Too few excitatory neurons"
assert diag['excitatory_fraction'] < 0.9, "Too many excitatory neurons"

result = jtfne.simulate(model, cfg.sim, backend='jaxfne')
```

---

## Advanced Topics

### Custom Receptor Kinetics (Future)

Currently, receptor specs are hardcoded from jaxfne.standard_receptor_specs.
Future versions will support custom tau_ms and reversal potentials via EdgeList
modification.

### Multi-Area Interactions (Future)

Currently, feedforward/feedback edges are defined spatially. Future versions will
support explicit inter-area routing with separate receptor types.

### Optimization & Plasticity (Future)

jaxfne's AGSDR optimizer can be integrated for network parameter tuning.

---

## References

- **jaxfne docs:** https://github.com/astuart/jaxfne
- **jbiophysic docs:** See `README.md` in repo root
- **JAX tutorial:** https://jax.readthedocs.io/en/latest/quickstart.html

---

## Citation

If you use the jaxfne backend in jbiophysic, please cite:

```bibtex
@software{jbiophysic2024,
  title={jbiophysic: Biophysical modeling and TFNE science-library kernels},
  version={1.1.1+jaxfne},
  url={https://github.com/HNXJ/jbiophysic}
}

@software{jaxfne2024,
  title={jaxfne: JAX-compiled forward-field and network emulator},
  url={https://github.com/astuart/jaxfne}
}
```

---

**Last Updated:** 2026-05-23  
**Status:** Complete, production-ready
