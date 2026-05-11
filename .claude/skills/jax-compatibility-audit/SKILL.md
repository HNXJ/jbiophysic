# JAX Compatibility Audit

## Purpose

Audit JAX compatibility before refactors. Ensure `jit`, `vmap`, `lax.scan`, `pmap`, device management, PRNG discipline, and NumPy/JAX crossings are correct. Preserve CPU-safe behavior without aggressively rewriting to `pjit` or `pmap` without tests.

## When to use

- Before modifying files in `src/jbiophysic/` that use JAX (cells, networks, optimization, simulation)
- Before proposing pmap/pjit/sharding changes
- Before touching PRNG key splitting or random state
- When investigating performance or device-count issues

## Inputs to inspect first

- `docs/jax_compatibility.md` (baseline: JAX 0.10.0, CPU-safe, 7 PRNG files identified)
- `P1_DECISIONS.md` (policy: preserve CPU-safe, no aggressive pmap rewriting)
- `src/jbiophysic/ops/gsgd.py` (device-count fallback pattern)
- `src/jbiophysic/ops/connectivity.py` and `edge_backend.py` (PRNG usage)

## Standard commands

```bash
# Find all JAX-related patterns
grep -RIn "jax\.pmap\|pmap\|pjit\|NamedSharding\|devices\|local_device_count\|random\.PRNGKey\|random\.key\|lax\.scan\|vmap\|jit" src tests

# Check device count behavior
grep -RIn "device_count\|num_devices" src tests

# Find NumPy/JAX crossings
grep -RIn "np\.\|numpy\." src/jbiophysic | grep -i "jnp\|jax" || echo "No obvious mixing detected"

# Run full test suite
source .venv/bin/activate
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=src python -m pytest -q --tb=short
```

## Safe procedure

1. Run grep patterns above to inventory JAX usage
2. For each file using pmap/pjit/devices:
   - Verify device-count fallback is present (e.g., in gsgd.py)
   - Verify PRNG keys are explicit, not global state
   - Check no silent device assumptions
3. For PRNG files (7 identified):
   - Verify explicit key splitting: `jax.random.split(key)`
   - Verify determinism test exists: same seed → same trajectory
4. For JIT/vmap:
   - Verify shapes are stable (no dynamic control flow in jit)
   - Verify vmap is only over batch dimensions
   - Verify no host callbacks in jitted code
5. Run full test suite
6. Validate no NaN/Inf in outputs (smoke gate)

## Validation gate

```bash
source .venv/bin/activate
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=src python -m pytest -q --tb=short
# Expected: 61 passed, 0 failed
```

If tests fail, do NOT proceed with JAX modernization. Fix the regression first.

## Stop conditions

- If any test fails
- If device fallback pattern is missing from pmap usage
- If PRNG keys are created globally or reused
- If NaN/Inf appears in simulation outputs
- If device-count check is removed without alternative fallback

## Final report fields

- PRNG files audited (should be 7)
- Device-count fallback present: yes/no
- jit/vmap patterns found: count and files
- pmap/pjit patterns found: count and files
- Determinism test coverage: covered/not covered
- Test baseline before: 61 passed
- Test baseline after: pass/fail
- Recommendation: safe to proceed / revise / blocked
