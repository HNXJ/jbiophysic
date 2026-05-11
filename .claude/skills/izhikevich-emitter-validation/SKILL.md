# Izhikevich Emitter Validation

## Purpose

Validate Izhikevich model changes. Preserve deterministic seed behavior, scan/JIT compatibility, explicit units, and no NaN/Inf. Treat parameter preset additions (RS/FS/chattering) and biological claim additions as API changes requiring tests and docs.

## When to use

- Before modifying `src/jbiophysic/cells/izhikevich.py`
- Before adding parameter presets or units documentation
- Before claiming biological realism or calibration to reference data
- After any changes to IzhikevichParams or simulation logic

## Inputs to inspect first

- `src/jbiophysic/cells/izhikevich.py` (scan-based, JIT-compatible)
- `CLAUDE.md` Section H (Izhikevich policy: preserve API, presets/docs later, no premature biological claims)
- `P1_DECISIONS.md` (Izhikevich policy: preserve API, add presets/tests later, exploratory not biological)
- Test files: `test_*.py` matching izhikevich

## Standard commands

```bash
# Locate Izhikevich code
find src -name "*.py" -exec grep -l "izhikevich\|Izhikevich" {} \;

# Check for parameter presets
grep -RIn "RS\|FS\|chattering\|preset\|parameter.*set\|neuron.*type" src/jbiophysic/cells/izhikevich.py || echo "No presets found"

# Check units in code
grep -RIn "mV\|ms\|pA\|nA\|uA\|Hz" src/jbiophysic/cells/izhikevich.py || echo "Units not explicitly documented in code"

# Run izhikevich-related tests
source .venv/bin/activate
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=src python -m pytest tests/ -k "izhikevich" -v --tb=short

# Run full suite
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=src python -m pytest -q --tb=short
```

## Safe procedure

1. Confirm test baseline passes (61/61)
2. Make changes to Izhikevich code
3. Run Izhikevich-specific tests first
4. Verify determinism: same seed → same voltage/spike trajectory
5. Verify no NaN/Inf in membrane potential, state variables, or outputs
6. Smoke gate: plausible voltage range check
   - v_m should stay in biological range (roughly -80 to +30 mV)
   - u should not diverge wildly
7. Run full test suite
8. If adding presets or units docs: confirm docs state "exploratory" not "validated"

## Validation gate

```bash
source .venv/bin/activate
# Determinism test
python - <<'PY'
import jax
import jax.numpy as jnp
from jbiophysic.cells.izhikevich import IzhikevichParams, izhikevich_scan

params = IzhikevichParams()
key = jax.random.PRNGKey(42)
key1, key2 = jax.random.split(key)

# Same seed, different key objects
v1 = izhikevich_scan(params, key1, 100)
v2 = izhikevich_scan(params, key1, 100)  # Same key

assert jnp.allclose(v1, v2), "Determinism failed: same seed should produce same trajectory"
print("Determinism PASS")
PY

# Full test suite
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=src python -m pytest -q --tb=short
# Expected: 61 passed, 0 failed
```

## Stop conditions

- If determinism test fails (same seed ≠ same trajectory)
- If NaN or Inf appears in membrane potential
- If voltage exceeds plausible range (< -100 mV or > +50 mV for extended periods)
- If any test fails
- If claims of biological calibration are added without reference benchmarks

## Final report fields

- Parameters modified: list
- Determinism verified: pass/fail
- NaN/Inf check: pass/fail
- Voltage range smoke gate: pass/fail
- Test baseline before: 61 passed
- Test baseline after: pass/fail
- API changes (presets/units): yes/no, if yes, docs added: yes/no
- Biological claim additions: yes/no, if yes, receipt-backed: yes/no
- Recommendation: safe to proceed / revise / blocked
