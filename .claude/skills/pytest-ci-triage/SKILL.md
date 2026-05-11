# pytest & CI Triage

## Purpose

Reproduce and classify local and GitHub Actions CI failures. Separate environment setup failure, dependency install failure, import failure, code syntax/lint failure, test failure, and timeout/hang. Fix one failure class per commit; do not mix CI compatibility fixes with style refactors.

## When to use

- When GitHub Actions CI fails
- When local tests fail unexpectedly
- When investigating "works locally but fails in CI" issues
- After any changes to dependencies, imports, or test setup
- Before committing changes that might affect CI

## Inputs to inspect first

- `.github/workflows/ci.yml` (Python matrix, install steps, lint/compile/test sequence)
- `CLAUDE.md` Section D (standard validation commands)
- `.venv/bin/activate` (local Python environment)
- Recent GitHub Actions logs: `gh run view <run-id> --log`

## Standard commands

```bash
# Confirm environment
cd /Users/hamednejat/workspace/main/jbiophysic
source .venv/bin/activate
python --version  # Should be 3.11.15
which python

# Clean install simulation (CI replicates this)
python -m pip install --upgrade pip
python -m pip install -e '.[dev,jax,tutorials]'

# CI steps in order
python - <<'PY'
import tomllib  # or handle 3.10 case
with open('pyproject.toml','rb') as f:
    print(tomllib.load(f)['project']['name'])
PY

ruff check src tests

PYTHONPATH=src python -m compileall -q src tests

PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=src python -m pytest -q --tb=short

# GitHub Actions monitoring
gh run list --limit 5
gh run view <run-id> --log
gh run view <run-id> --log-failed
```

## Safe procedure

1. Run local CI reproduction steps in order (environment → install → pyproject → lint → compile → test)
2. Identify first failure point
3. Classify failure:
   - **Environment**: Python version mismatch, venv not activated
   - **Install**: `pip install -e '.[dev]'` fails with missing/conflicting deps
   - **Pyproject**: tomllib/tomli import fails (see `ci-parse-fix` if on Python 3.10)
   - **Lint**: `ruff check src tests` reports style violations
   - **Compile**: `py_compile` syntax errors
   - **Import**: module not found or circular import
   - **Test**: test assertion failures, test setup failures
   - **Timeout/Hang**: step takes >30s or never completes
4. If local success but CI fails: suspect matrix differences (Python 3.10, 3.11, 3.12 have different behavior)
5. For each failure class, create a separate commit with focused fix
6. Do NOT combine ruff style fixes with CI compatibility fixes

## Validation gate

```bash
source .venv/bin/activate
python --version
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=src python -m pytest -q --tb=short
# Expected: 61 passed, 0 failed
```

After fix commit:

```bash
git push origin main
sleep 5
gh run list --limit 2
# Monitor until new run completes (success or next failure class)
```

## Stop conditions

- If multiple failure classes exist, fix ONE and stop
- If a fix introduces a new regression (e.g., fixes lint but breaks test), revert and diagnose
- If CI is still progressing past the fixed step (i.e., fix worked), prepare next commit for next failure class
- If CI fix requires modifying source logic (not just imports or setup), verify full test suite passes locally first

## Failure classification quick reference

| Failure point | Root cause examples | Fix examples |
|---|---|---|
| Install | Missing dep, version conflict | `pip install <package>`, update pyproject.toml |
| Pyproject parse | `tomllib` not in Python 3.10 | Fallback to tomli (see ci-parse-fix skill) |
| Lint (ruff) | E501 line too long | Refactor lines, or add --ignore flag (band-aid) |
| Compile (py_compile) | Syntax errors, unresolved imports | Fix syntax, add imports |
| Test | Assertion failure, setup error | Debug test, fix code or test fixture |
| Timeout | Infinite loop, very slow operation | Profile, optimize, or add timeout |

## Final report fields

- CI failure point: (step name and error)
- Failure classification: (environment/install/import/lint/test/timeout)
- Root cause: (specific error message or condition)
- Fix applied: (specific change)
- Local reproduction: (pass/fail)
- CI run URL after fix: (GitHub Actions link)
- Next failure class (if any): (or "CI passing")
- Recommendation: (next action - another fix commit, or CI clean)
