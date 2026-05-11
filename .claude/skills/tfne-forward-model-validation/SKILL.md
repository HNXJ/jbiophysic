# TFNE Forward Model Validation

## Purpose

Validate TFNE (Emitter → Source → Field → Probe) work. Keep TFNE as a forward field model for CSD/LFP, not a whole-brain theory. Require source conservation, gauge/mean-zero checks, SPD/passivity validation, and no NaN/Inf. Preserve the architecture; do not claim biophysical accuracy without empirical validation.

## When to use

- Before modifying `src/jbiophysic/tfne/` (sources, fields, solvers)
- Before adding source conservation, gauge, SPD, or passivity tests
- Before claiming whole-brain simulation or conductivity calibration
- After changes to extracellular/resistive forward model

## Inputs to inspect first

- `docs/tutorial_status.md` (TFNE scope: forward-field tool, not whole-brain)
- `docs/jax_compatibility.md` (TFNE mentioned in scope)
- `P1_DECISIONS.md` (TFNE policy: preserve Emitter→Source→Field→Probe, add conservation/gauge/SPD tests later)
- `src/jbiophysic/tfne/sources.py`, `fields.py`, `solvers.py`

## Standard commands

```bash
# Locate TFNE code
find src -path "*/tfne/*" -name "*.py" -exec ls -l {} \;

# Check for conservation/gauge/SPD/passivity tests
grep -RIn "conservation\|gauge\|mean.*zero\|SPD\|passivity\|positive.*definite" src/jbiophysic/tfne tests || echo "No conservation/gauge tests found (expected, add later)"

# Check for NaN/Inf detection
grep -RIn "nan\|inf\|isnan\|isinf" src/jbiophysic/tfne || echo "No explicit NaN/Inf checks found"

# Run TFNE-related tests
source .venv/bin/activate
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=src python -m pytest tests/ -k "tfne" -v --tb=short

# Run full suite
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=src python -m pytest -q --tb=short
```

## Safe procedure

1. Confirm test baseline passes (61/61)
2. Make changes to TFNE code (sources, fields, solvers)
3. Run TFNE-specific tests first
4. Verify source integral (conservation):
   - Total emitter current should equal total source integral (within declared tolerance)
   - Document tolerance: `assert_conservation_tolerance = <value>`
5. Verify gauge (mean-zero if resistive):
   - If using resistive model, verify mean of field is near zero
   - If using anisotropic/bidomain, document gauge choice
6. Check for NaN/Inf:
   - No NaN in field values
   - No Inf in probe outputs
   - All conductivity/resistivity values positive and finite
7. Smoke gate: plausible extracellular potential range
   - Extracellular potential should stay in reasonable range (±10 mV typical for LFP)
8. Run full test suite

## Validation gate

```bash
source .venv/bin/activate
# Source conservation check (if applicable)
python - <<'PY'
import jax.numpy as jnp
from jbiophysic.tfne.sources import /* source class */
from jbiophysic.tfne.fields import /* field solver */

# Create test source
source = /* instantiate */
field = /* solve field from source */

# Check conservation
source_integral = jnp.sum(source)  # or appropriate integral
field_integral = jnp.sum(field)  # or line integral

conservation_error = jnp.abs(source_integral - field_integral)
assert conservation_error < 1e-6, f"Conservation failed: {conservation_error}"
print("Source conservation PASS")
PY

# Full test suite
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=src python -m pytest -q --tb=short
# Expected: 61 passed, 0 failed
```

## Stop conditions

- If any test fails
- If field contains NaN or Inf
- If source integral diverges wildly (beyond declared tolerance)
- If extracellular potential exceeds plausible range persistently
- If whole-brain or conductivity calibration claims are added without receipts
- If gauge/mean-zero property is broken without explicit documentation

## Final report fields

- TFNE files modified: list
- Architecture preserved (Emitter→Source→Field→Probe): yes/no
- Source conservation tolerance: declared/not declared, if declared, value: X
- Gauge/mean-zero property: tested/not tested, if tested, status: pass/fail
- SPD/passivity tests: added/not added
- NaN/Inf check: pass/fail
- Extracellular potential range: plausible/implausible
- Test baseline before: 61 passed
- Test baseline after: pass/fail
- Biological accuracy claims added: yes/no, if yes, receipts provided: yes/no
- Recommendation: safe to proceed / revise / blocked
