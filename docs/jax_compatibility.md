# JAX Compatibility and Performance Policy

**Status:** Baseline validated, modernization staged  
**Date:** 2026-05-10  
**Truth mode:** exploratory, not biological validation

---

## Current Baseline (P0)

- **JAX version:** 0.10.0
- **Environment:** Python 3.11.15, CPU device (single CpuDevice(id=0))
- **Test status:** 61/61 passing, deterministic
- **PRNG status:** Explicit key discipline; same seed → same result

---

## JAX Compatibility Scope

### What's in scope:
- CPU-safe execution
- Deterministic PRNG with explicit key splitting
- JAX arrays and pytrees
- vmap/scan/jit where currently used
- No host callbacks or side effects in jitted code

### What's NOT in scope yet:
- pmap/pjit/sharding modernization (staged for separate phase)
- xmap or shard_map rewrites (not blocking)
- Over-jitting micro-optimizations
- GPU/TPU testing (CPU-only baseline)

---

## Device-Count Fallback Behavior

**gsgd.py pattern (current, safe for CPU):**

```python
num_devices = jax.device_count()
if num_devices > 1:
    # pmap + vmap for multi-device
    losses = jax.pmap(jax.vmap(loss_fn_single))(reshaped_pop)
else:
    # vmap-only fallback for single device (CPU)
    losses = jax.vmap(loss_fn_single)(population)
```

**Policy:**
- Multi-device logic is preserved.
- CPU fallback is safe and tested.
- Do not aggressively rewrite to pjit without tests.

---

## PRNG Discipline

**Current files using PRNG (7):**
- gsgd.py
- gsdr.py
- connectivity.py (networks)
- edge_backend.py (simulation)
- test_legacy_pipeline.py
- test_optim_network_pipeline.py
- audit_repo.py (script)

**Policy:**
- Explicit keys, no global mutable state
- Key splitting via `jax.random.split()` inside loops
- Determinism test: same seed → same trajectory
- No silent PRNG reuse

---

## Optax Compatibility

**Status:** Optional JAX-extra dependency  
**Version:** 0.2.8  
**Core import requirement:** NOT required for core jbiophysic package

**Current usage:** 0 (Optax is declared but not yet integrated)

**Policy:**
- Keep Optax optional; do not force core dependency
- GSDR/AGSDR may use custom optimization; no forced rewrite
- Optional adapter wrapper may be added later with guarded imports
- Example pattern:
  ```python
  try:
      import optax
      has_optax = True
  except ImportError:
      has_optax = False
  ```

---

## Modernization Roadmap (Not blocking P0/P1)

### Phase 2 (Future): pmap/pjit audit
- Document device-count fallback behavior
- Add CPU-only tests
- Create `src/jbiophysic/ops/parallel.py` compatibility layer
- Test multi-device patterns if GPU/TPU available

### Phase 3 (Future): pjit + NamedSharding migration
- Replace pmap with pjit + PartitionSpec
- Requires tests for multi-device environments
- Optional; CPU fallback is production-ready

### Phase 4 (Future): PRNG modernization
- Audit all 7 files for determinism
- Add `@jax.deterministic_key_seed` helpers if needed
- Document explicit key discipline

---

## Testing and Validation

**Required before modernization:**
- Green baseline (61/61 tests) ✅
- CPU-only environment ✅
- Deterministic PRNG tests (pending)
- No regressions in execution time (pending)

**Not required for P0/P1:**
- Multi-device testing
- GPU/TPU validation
- Performance benchmarks
