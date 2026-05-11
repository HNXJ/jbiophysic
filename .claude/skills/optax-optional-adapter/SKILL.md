# Optax Optional Adapter

## Purpose

Guide future Optax integration without making it a forced core dependency. Keep Optax optional under the [jax] extra. Do not rewrite GSDR/AGSDR around Optax without explicit approval. Any adapter must use guarded imports and tests. Core `import jbiophysic` must work without Optax if JAX extras are not installed.

## When to use

- Before writing any Optax-dependent code
- Before modifying GSDR/AGSDR to use Optax
- Before adding Optax to core imports (not allowed)
- When considering whether Optax integration is appropriate
- When creating an optional Optax adapter/wrapper

## Inputs to inspect first

- `pyproject.toml` (dependencies: Optax is in [jax] extra, version 0.2.8)
- `P1_DECISIONS.md` (Optax policy: keep optional, no core rewrite, optional adapter later with guarded imports)
- `CLAUDE.md` Section G (Optax compatibility: keep optional, no mandatory core import)
- `src/jbiophysic/ops/gsgd.py` and `gsdr.py` (current custom optimization)

## Standard commands

```bash
# Check current Optax usage
grep -RIn "import optax\|from optax" src tests

# Verify core import works without [jax]
python - <<'PY'
# Simulate base install (no [jax])
import sys
try:
    import optax
    print("ERROR: optax should NOT be importable in base install")
    sys.exit(1)
except ImportError:
    print("OK: optax not available in base install (expected)")

# But jbiophysic should still import
import jbiophysic
print("OK: jbiophysic imports without optax")
PY

# Verify [jax] install makes Optax available
pip install -e '.[jax]'
python - <<'PY'
import optax
print(f"OK: optax {optax.__version__} available with [jax]")
PY

# Test guarded import pattern
python - <<'PY'
try:
    import optax
    HAS_OPTAX = True
except ImportError:
    HAS_OPTAX = False

if HAS_OPTAX:
    print(f"Optax available: {optax.__version__}")
else:
    print("Optax not available; using fallback")
PY
```

## Safe procedure

1. **Before writing any Optax code:**
   - Confirm core `import jbiophysic` works WITHOUT [jax] extra
   - Confirm guarded import pattern compiles

2. **If creating an optional adapter:**
   - Put adapter in separate module: `src/jbiophysic/ops/optax_adapter.py` (example)
   - Use guarded imports at module level:
     ```python
     try:
         import optax
         HAS_OPTAX = True
     except ImportError:
         HAS_OPTAX = False
     ```
   - If HAS_OPTAX is False, raise clear error (not obscure AttributeError):
     ```python
     if not HAS_OPTAX:
         raise ImportError("optax required; install jbiophysic[jax]")
     ```
   - Test both paths: with Optax and without

3. **Do NOT:**
   - Add Optax to core imports in `src/jbiophysic/__init__.py`
   - Rewrite GSDR/AGSDR without approval (they have working custom optimization)
   - Force users to install [jax] extra to use jbiophysic (base install should work)

4. **Test coverage:**
   - Test base install works: `pip install -e '.'` then `import jbiophysic`
   - Test [jax] install works: `pip install -e '.[jax]'` then adapter loads
   - Test adapter failure is clear: try to use adapter without [jax], should raise ImportError with message

## Validation gate

```bash
source .venv/bin/activate

# Confirm core import works (do this in clean venv with only base install)
python - <<'PY'
import jbiophysic
print("✓ Core import OK")
PY

# Confirm [jax] install makes Optax available
pip install -e '.[jax]'
python - <<'PY'
import optax
import jbiophysic
print("✓ Optax available with [jax]")
PY

# Run tests (should all pass)
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=src python -m pytest -q --tb=short
# Expected: 61 passed, 0 failed
```

## Stop conditions

- If core `import jbiophysic` fails without [jax]
- If Optax is added to core imports
- If GSDR/AGSDR is rewritten without approval
- If adapter error message is obscure (must clearly say "install jbiophysic[jax]")
- If adapter tests are missing or fail
- If any test fails

## Final report fields

- Optax usage in code: (none / adapter only / core import error)
- Core import test (without [jax]): (pass/fail)
- [jax] install test (with Optax): (pass/fail)
- Guarded import pattern used: (yes/no)
- Adapter module location: (none / file path)
- Adapter error message clarity: (clear / obscure)
- GSDR/AGSDR modifications: (none / approval obtained)
- Test coverage: (pass/fail)
- Recommendation: (safe to proceed / revise / blocked)
