# jbiophysic Claude Code Operating Context

**Status:** Gamma Labyrinth alignment (jbiophysic specialized).  
**Date:** 2026-05-10  
**Checkpoint:** commit aaa83cbf69eaac0a9b6a1fe3e540cfc04833a94d  
**truth_mode:** truth_safe_unverified, tutorial_exploratory_not_biological_truth

---

## A. Identity and Report Rule

Every Claude Code report for jbiophysic must begin and end with:

```
[model-llm-name][/Users/hamednejat/workspace/main/jbiophysic][yyyymmdd-hhmm]
```

**Rules:**
- Do not guess model identity or workspace path; use verified values.
- Use `pwd` and `python --version` in active shell to confirm.
- If unavailable, use: `[unknown_model_do_not_guess][jbiophysic][yyyymmdd-hhmm]`
- Include start-of-work and end-of-work timestamps.

---

## B. Repo Role and Main Themes

**jbiophysic** is an exploratory computational biophysics / computational neuroscience research infrastructure:

- **Scope:** Izhikevich and HH-style neuron models, laminar E/PV/SST/VIP cortical circuits, multi-area hierarchy simulations, global oddball/omission task scaffolds, TFNE forward-field CSD/LFP modeling, optimization/plasticity experiments.
- **Plane:** Execution (code, tests, tutorials, docs as first-class artifacts).
- **Truth status:** Exploratory research infrastructure. No biological/scientific claims without receipt-backed validation. Optimizer success ≠ biological proof.

**Main active themes:**

| Theme | Role | Status |
|-------|------|--------|
| JAX-compatible neuron/circuit sim | Core computation framework | JAX 0.10.0, CPU-safe, baseline 96/96 tests |
| Izhikevich emitters | Spiking neuron model | Preserve API; presets/docs/tests later |
| Hodgkin-Huxley emitters | Biophysical neuron model | Preserve API; integrate with HH tests |
| TFNE: Emitter → Source → Field → Probe | Forward-field CSD/LFP framework | Preserve architecture; add conservation/gauge tests later |
| GSDR/AGSDR optimization | Fitness/evolution experiments | Keep custom optimization; Optax optional |
| Tutorials (nbconvert-portable) | Executable teaching artifacts | 5 main notebooks; Colab artifacts archived separately |
| Docs | Policy and guardrails | Scientific truth status clearly marked |

---

## C. Current Verified Baseline

**Last validated checkpoint:** commit aaa83cbf69eaac0a9b6a1fe3e540cfc04833a94d  
**Date verified:** 2026-05-10

**Environment:**
```bash
cd /Users/hamednejat/workspace/main/jbiophysic
source .venv/bin/activate
python --version  # Python 3.11.15
```

**Installation command (verified):**
```bash
python -m pip install -e '.[dev,tutorials]'
```
Note: JAX is now core (required); [jax] extra is no longer needed.

**Test baseline:**
```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=src python -m pytest -q --tb=short
# Result: 96 passed, 0 failed, 0 errors, 2 non-critical warnings (~18s)
```

**JAX/Optax imports (verified):**
- `jax.__version__` → 0.10.0
- `jax.devices()` → [CpuDevice(id=0)]
- `optax.__version__` → 0.2.8
- `jaxlib.__version__` → 0.10.0 (via jax)

**Focused module imports (verified):**
```python
import jbiophysic  # OK
import jbiophysic.cells.izhikevich  # OK
import jbiophysic.cells.hh  # OK
import jbiophysic.tfne  # OK
import jbiophysic.tfne.sources  # OK
import jbiophysic.tfne.fields  # OK
import jbiophysic.tfne.solvers  # OK
```

**pyproject.toml constraints:**
- Python >=3.10 required
- Core deps: numpy, scipy, pandas, PyYAML, jax, jaxlib, equinox, optax, diffrax
- Tutorials extra: jupyter, nbformat, nbconvert, ipykernel, matplotlib (optional)
- Dev extra: pytest, pytest-cov, ruff, black (optional)

---

## D. Standard Validation Commands

**Before every commit/push, run:**

```bash
# 1. Activate venv
source .venv/bin/activate

# 2. Run full test suite
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=src python -m pytest -q --tb=short

# 3. Syntax check any modified .py files (example):
python -m py_compile src/jbiophysic/cells/izhikevich.py

# 4. No-secrets scan (see Section J)
grep -RInE 'api[_-]?key|secret|token|password|private[_-]?key|bearer|BEGIN .*PRIVATE KEY' src/ tests/ docs/ .claude 2>/dev/null || true
```

**After rebase/merge, re-run tests:**

```bash
git pull --rebase origin main
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=src python -m pytest -q --tb=short
# If any failure, do NOT push; fix locally first.
```

**Notebook validation principles:**

- Portable notebooks: source cells parse via `"".join(cell["source"])` and execute via `nbconvert --execute`
- Colab artifacts: labeled with `.colab.ipynb` suffix; contain `google.colab`, `%cd`, `!pip` magic commands
- Execution counts: sequential, not mixed
- Outputs: present and up-to-date; do NOT commit outdated outputs

---

## E. Development Policy

**Git discipline:**

- No broad `git add .` ever.
- Stage exact file paths only: `git add README.md docs/jax_compatibility.md`
- Verify staging: `git diff --cached --name-only` before commit
- Tests must pass before commit and after rebase before push
- No force push ever (`--force`, `--force-with-lease`)
- Branch deletion postponed; see Section K

**Workflow:**

1. Confirm clean starting state: `git status --short --branch`
2. Make changes in focused scope (no sprawl)
3. Run validation (Section D)
4. Stage exact files only
5. Commit with clear message: `git commit -m "category: brief message"`
6. Fetch and rebase: `git fetch origin && git pull --rebase origin main`
7. Re-run tests
8. Push: `git push origin main`
9. If any step fails, diagnose and fix locally before retrying; never force

**Scope discipline:**

- Do not refactor broadly in one commit
- Do not delete without grep/import/reference audit + manifest
- Do not rewrite APIs without tests
- Do not integrate Optax into core code without separate approval
- Do not modernize pmap/pjit without baseline tests
- GAMMA phases are narrow and sequential; no phase parallelization

---

## F. JAX Policy

**Current baseline:** JAX 0.10.0, CPU-safe, vmap/scan/jit tested, pmap with fallback.

**Preserve:**
- CPU-safe execution (do not assume GPU)
- Device-count fallback pattern in gsgd.py: `if num_devices > 1: pmap(...) else: vmap(...)`
- Deterministic PRNG with explicit key splitting
- vmap for batch dimensions (shape-stable only)
- jit for stable shapes and tested functions
- lax.scan for recurrent computations

**Do not (without separate approval):**
- Rewrite pmap to pjit aggressively
- Use shard_map or xmap without tests
- Add over-aggressive jit decorators (micro-optimization)
- Assume multi-device/GPU behavior without testing

**PRNG discipline (7 files identified):**

Files using PRNG: `gsgd.py`, `gsdr.py`, `connectivity.py` (networks), `edge_backend.py` (simulation), `test_legacy_pipeline.py`, `test_optim_network_pipeline.py`, audit scripts.

Policy:
- Explicit keys via `jax.random.split()`, no global mutable state
- Same seed → same deterministic trajectory (test this)
- No silent PRNG reuse
- Document seed scope in code comments if non-obvious

**NumPy/JAX crossings:**
- Keep intentional (do not scatter JAX/NumPy calls randomly)
- Use JAX arrays for autodiff/JIT
- Convert to NumPy for I/O/visualization only
- Avoid repeated host-to-device transfers in loops

---

## G. Optax Policy

**Current status:** Optax 0.2.8 available in [jax] extra; zero current uses in core code.

**Policy:**
- Keep Optax as optional JAX-extra dependency
- Do NOT rewrite GSDR/AGSDR around Optax without separate approval
- Core imports must not require Optax (guard with try/except if ever used)
- Any future adapter must be tested and optional

**Example guarded import (if needed later):**

```python
try:
    import optax
    HAS_OPTAX = True
except ImportError:
    HAS_OPTAX = False

if HAS_OPTAX:
    # Use optax
    pass
else:
    # Use custom optimization
    pass
```

---

## H. Izhikevich Policy

**Current API:** IzhikevichParams, scan-based simulation, JIT-compatible.

**Preserve:**
- Existing parameter API and tests (61/61 baseline)
- Scan/JIT compatibility
- Deterministic seeds
- Unit consistency (ms, mV, pA/nA where applicable)

**Do not claim yet:**
- Biological calibration without benchmarking
- Physiological realism without empirical validation
- Spike timing fidelity beyond simulation precision

**Add later (separate approval):**
- Parameter presets (RS/FS/chattering neurons)
- Unit validation docs
- Deterministic seed tests
- Biophysical parameter bounds

**Truth status:** Exploratory tutorial; not validated against reference neuroscience data.

---

## I. TFNE Policy

**Current architecture:** Emitter → Source → Field → Probe

**Preserve:**
- Forward-field framing (CSD/LFP modeling)
- Source/field decoupling
- Probe geometry independence

**Do not claim:**
- Whole-brain simulation capability
- Biophysical accuracy without conservation/gauge/SPD validation
- Conductivity model beyond resistive

**Add later (separate approval):**
- Source conservation tests with tolerance specs
- Mean-zero gauge validation (current is conserved and sink-free)
- Symmetric Positive Definite (SPD) tensor validation
- NaN/Inf detection and rejection
- Passivity/causality tests

**Truth status:** Forward-field tool; exploratory framework; not a whole-brain simulator.

---

## J. Tutorial / Docs Policy

**Portable tutorials (nbconvert-runnable, no magic):**
- `tutorials/00_neuronal_equations_book.ipynb` (equation families)
- `tutorials/01_izhikevich_hh_single_neurons.ipynb` (single-neuron models)
- `tutorials/02_tfne_forward_fields.ipynb` (forward-field modeling)
- `tutorials/03_tfne_izhikevich_hybrid.ipynb` (Izhikevich-to-TFNE)
- `tutorials/04_laminar_oddball_three_area_cortex.ipynb` (laminar cortex)

**Colab artifacts (archived, not portable):**
- `tutorials/source_notebooks/tfne_izhikevich_net.colab.ipynb` (labeled .colab; contains google.colab, %cd, !pip)

**Documentation:**
- `docs/jax_compatibility.md` — JAX baseline, scope, device fallback, PRNG, Optax, modernization roadmap
- `docs/tutorial_status.md` — Notebook classification, audit evidence, scientific guardrails
- `README.md` — Quick start, install modes, baseline evidence, truth status

**Rules:**
- All docs must explicitly state truth status: "exploratory", "not biological validation", "teaching artifact"
- Tutorials are educational scaffolds; do not claim biological calibration
- Scientific guardrails section: state scope limits (Izhikevich native current, TFNE forward-field, oddball scaffolds illustrative)
- Links to external benchmarks/validation if available

---

## K. No-Secrets Policy

**Never commit, document, or log:**
- API keys, bearer tokens, passwords, private keys
- OAuth tokens, SSH credentials, session tokens
- Supabase/LM Studio keys, AWS/GCP credentials, database passwords
- .env file contents (actual values)

**Before every commit:**

```bash
grep -RInE 'api[_-]?key|secret|token|password|private[_-]?key|bearer|BEGIN .*PRIVATE KEY' src/ tests/ docs/ .claude 2>/dev/null || true
```

If grep finds real secrets:
1. Stop immediately; do not commit
2. Remove/redact the value
3. Replace with `[REDACTED_SECRET_LIKE_VALUE]`
4. Commit the redacted version
5. Report with identity header/footer

If grep finds only policy text (e.g., "password policy"), report safe.

---

## L. Branch Policy and Deletion

**Default:** main is source-of-truth.

**Before branch-sensitive work:**
```bash
git status --short --branch
git fetch origin
git rev-parse origin/main
```

**No force push ever.**

**Branch deletion (final administrative phase only):**
- Deletion is NOT part of early GAMMA
- Requires after-GAMMA-completion when all tests pass
- Requires grep/import audit of code references
- Requires manifest of branch SHAs: `git ls-remote --heads origin | sort > BRANCH_HEADS_ARCHIVE.txt`
- Requires archival tags: `git tag -a archive/branch-name-YYYYMMDD <sha>`
- Requires explicit user confirmation: `CONFIRM_DELETE_NON_MAIN_BRANCHES=YES`
- Only then: `git push origin --delete branch-name`

**Current non-main branches (tracked but not deleted):**
- None actively developed in jbiophysic; main is the only active branch

---

## M. Final Report Template

Every Claude Code session ends with this format:

```
[claude-haiku-4-5-20251001][/Users/hamednejat/workspace/main/jbiophysic][yyyymmdd-hhmm]

### Work Performed
- Files changed: [list]
- Repo: jbiophysic
- Branch: main
- HEAD before: [sha]
- HEAD after: [sha]
- Commands run: [list]

### Validation
- Tests before: [status]
- Tests after: [status]
- No-secrets scan: [safe/redacted]
- Staged files: [list of exact paths]
- Commit SHA: [sha]

### Evidence
- Commit/PR link: https://github.com/HNXJ/jbiophysic/commit/[sha]
- Push result: [git output]

### Risks and Blockers
- [None or specific issues]

### Truth Status
- truth_mode: truth_safe_unverified
- truth_bearing_run: false (unless new validated evidence produced)

### Next Safe Action
[One specific, actionable step]

[claude-haiku-4-5-20251001][/Users/hamednejat/workspace/main/jbiophysic][yyyymmdd-hhmm]
```

**Rules:**
- Do not invent runtime state
- Do not hide risks
- Do not claim success without validation
- Do not assert truth without receipts

---

## N. Quick Start for Future Sessions

**Confirm baseline:**
```bash
cd /Users/hamednejat/workspace/main/jbiophysic
source .venv/bin/activate
python --version  # Should be 3.11.15
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=src python -m pytest -q --tb=short
# Expected: 61 passed, 0 failed
```

**Standard workflow:**
1. Read this file (Section N onwards)
2. Read `README.md` for repo overview
3. Read `docs/jax_compatibility.md` and `docs/tutorial_status.md` for current policy state
4. Inspect `P0_BASELINE_SUMMARY.md` and `P1_DECISIONS.md` for audit trail
5. Consult `AUDIT_AND_REFACTOR_PLAN.md` for approved GAMMA phases
6. Work in narrow scope (one GAMMA phase, one theme)
7. Run validation before every commit
8. End with identity-wrapped report (Section M)

---

## O. Related Files

- `README.md` — Install modes, quick validation, baseline evidence, tutorial classification
- `docs/jax_compatibility.md` — JAX baseline and modernization roadmap
- `docs/tutorial_status.md` — Notebook audit evidence and scientific guardrails
- `P0_BASELINE_SUMMARY.md` — Baseline validation evidence (Python 3.11, 61 tests, imports)
- `P1_DECISIONS.md` — Policy decisions (Optax, Colab, pmap, Izhikevich, TFNE, legacy)
- `AUDIT_AND_REFACTOR_PLAN.md` — Approved GAMMA phases and scope definition
- `/Users/hamednejat/.claude/CLAUDE.md` — Global Gamma Labyrinth doctrine (reference only)

---

**Status:** jbiophysic Claude Code context finalized for GAMMA Phase 2 and beyond.  
**Last updated:** 2026-05-10  
**truth_mode:** truth_safe_unverified
