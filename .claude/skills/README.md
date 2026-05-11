# Claude Code Skills for jbiophysic

**Purpose:** Operating guides for Claude Code agents working on jbiophysic. These skills reduce drift, accelerate decision-making, and ensure safety gates are applied consistently.

**Scope:** Documentation and validated procedures only. Skills do not modify source code; they guide how to audit, validate, and safely make changes.

---

## Skill Index

### 1. JAX Compatibility Audit
📍 **File:** `jax-compatibility-audit/SKILL.md`

**Use when:** Modifying JAX code (jit, vmap, pmap, PRNG, device handling, NumPy/JAX crossings)

**Gate:** Verify device-count fallback, explicit PRNG keys, no hidden global state, determinism test passes, full test suite passes.

**Policy:** Preserve CPU-safe behavior. Do not aggressively rewrite pmap→pjit without tests.

---

### 2. Izhikevich Emitter Validation
📍 **File:** `izhikevich-emitter-validation/SKILL.md`

**Use when:** Modifying `src/jbiophysic/cells/izhikevich.py`, adding presets/units, or claiming biological behavior

**Gate:** Determinism test (same seed → same trajectory), no NaN/Inf, plausible voltage range, no biological claims without empirical validation.

**Policy:** Preserve API and scan/JIT compatibility. Treat presets and biological claims as API changes requiring docs and receipts.

---

### 3. TFNE Forward Model Validation
📍 **File:** `tfne-forward-model-validation/SKILL.md`

**Use when:** Modifying TFNE source/field/probe code, adding conservation/gauge/SPD tests, or claiming biophysical accuracy

**Gate:** Source conservation (declare tolerance), gauge/mean-zero check, no NaN/Inf, plausible extracellular potential range, full test suite passes.

**Policy:** Preserve Emitter→Source→Field→Probe architecture. TFNE is a forward model, not whole-brain theory.

---

### 4. pytest & CI Triage
📍 **File:** `pytest-ci-triage/SKILL.md`

**Use when:** GitHub Actions CI fails, local tests fail, or investigating CI vs local discrepancies

**Gate:** Classify failure (environment / install / parse / lint / test / timeout), fix ONE class per commit, verify local reproduction.

**Policy:** Fix CI compatibility issues separate from style refactors. One failure class = one commit.

---

### 5. Notebook Artifact Integrity
📍 **File:** `notebook-artifact-integrity/SKILL.md`

**Use when:** Committing notebook changes, creating tutorials, archiving Colab artifacts, or notebook parse/execution fails

**Gate:** AST parse (no syntax errors), execution counts sequential, outputs present/fresh, Colab artifacts labeled/archived separately.

**Policy:** Portable notebooks (00-04) are nbconvert-executable. Colab artifacts (.colab.ipynb) are reference-only. Source notebooks are WIP.

---

### 6. Optax Optional Adapter
📍 **File:** `optax-optional-adapter/SKILL.md`

**Use when:** Creating Optax integration, modifying GSDR/AGSDR, or adding Optax-dependent code

**Gate:** Core `import jbiophysic` works without [jax] extra. Adapter uses guarded imports. Failure messages are clear ("install jbiophysic[jax]").

**Policy:** Keep Optax optional [jax] extra. Do not rewrite GSDR/AGSDR without approval. No forced core dependency.

---

### 7. Legacy Code Cleanup Manifest
📍 **File:** `legacy-cleanup-manifest/SKILL.md`

**Use when:** Deleting dead code, removing deprecated modules, archiving experimental code

**Gate:** Complete deletion manifest (candidate, reason, grep audit results, rollback method), zero references found, test suite passes after deletion.

**Policy:** No deletion without manifest. Grep/import audit required. Delete ONE thing per commit. Document rollback.

---

## Quick Reference: Which Skill?

| Situation | Skill |
|-----------|-------|
| Modifying JAX imports, pmap, vmap, jit, PRNG, or device logic | **1. JAX Audit** |
| Changing Izhikevich parameters, adding presets, or making biological claims | **2. Izhikevich** |
| Modifying TFNE sources/fields/solves or claiming biophysical accuracy | **3. TFNE** |
| GitHub Actions CI fails, ruff lint fails, or tests fail | **4. pytest & CI Triage** |
| Creating/updating/archiving notebooks | **5. Notebook Integrity** |
| Adding Optax code or rewriting optimization | **6. Optax Adapter** |
| Deleting files, modules, or dead code | **7. Legacy Cleanup** |

---

## Standard Workflow

1. **Before making changes:** Identify which skill(s) apply
2. **While making changes:** Follow the skill's "Safe procedure" section
3. **Before committing:** Validate via the skill's "Validation gate"
4. **If any gate fails:** Do not commit; revise per skill's "Stop conditions"
5. **Commit message:** Reference the skill (e.g., "feat(jax): update vmap pattern per jax-audit skill")
6. **Final report:** Include skill's "Final report fields" in commit or PR description

---

## Truth Status

All skills maintain the repo's truth doctrine:
- **truth_safe_unverified:** No biological/scientific claims without receipt-backed validation
- **exploratory_not_production:** Code is research infrastructure, not production system
- **tutorial_not_truth:** Tutorials are teaching artifacts; optimizer success ≠ biological proof

---

## Related Documentation

- `CLAUDE.md` — Repo-local operating context (identity, baseline, policy)
- `P0_BASELINE_SUMMARY.md` — P0 validation (Python 3.11, 61 tests, imports)
- `P1_DECISIONS.md` — Policy decisions (Optax, Colab, pmap, branches)
- `.github/workflows/ci.yml` — CI matrix and steps
- `docs/jax_compatibility.md` — JAX baseline and modernization roadmap
- `docs/tutorial_status.md` — Notebook classification and guardrails

---

**Last updated:** 2026-05-11  
**Status:** 7 skills finalized for jbiophysic  
**Used by:** Claude Code agents on GAMMA phases 2–8
