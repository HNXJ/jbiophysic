# jbiophysic v0.0: Atlas Identity & Package Hygiene

**Status:** v0.0.0–v0.0.11 COMPLETE  
**Date:** 2026-05-23  
**Scope:** Transform jbiophysic from "large experimental codebase" to navigable biophysical atlas

---

## v0.0.0: Identity Doctrine

**jbiophysic is an executable biophysics atlas.**

It teaches:
- Membrane capacitance, leak, RC constants, and driving force
- HH channel kinetics, gating, time constants, reversal potentials
- Izhikevich reductions and what they omit
- Synaptic receptor kinetics (AMPA, NMDA, GABA-A, GABA-B)
- E/I balance, laminar cell-type priors
- CSD/LFP volume conduction assumptions
- Sensory, omission, and oddball case studies
- Nulls, ablations, synchrony gates
- Manifest-grounded interpretation

**jbiophysic uses jaxfne as a backend only:**
- jaxfne is imported optionally via [jaxfne] extra
- jaxfne's source-to-field/readout contract is used when chapters reach explicit Emitter→Source→Field→Probe semantics
- This happens in phase .7 of each chapter (e.g., v0.3.7, v0.5.7, v0.6.7)

**NOT BIOLOGICAL PROOF:**
- Optimizer success is not mechanism proof
- Native Izhikevich currents are not SI units without calibration
- TFNE source-field solutions are computational proxies, not validated biophysics
- All claims marked with truth_safe_unverified and claim-gate contracts

---

## v0.0.1: Version Reconciliation

✅ **COMPLETE**

- pyproject.toml is single source of truth: version = "1.0.1"
- __init__.py derives __version__ from importlib.metadata
- Fallback to "1.0.1" if package not installed (dev mode)

---

## v0.0.2: Optional Dependency Policy

✅ **COMPLETE**

Install modes:

```toml
[project.optional-dependencies]
jax = [jax, jaxlib, equinox, optax, diffrax]
jaxfne = [jaxfne==0.2.30]
jaxley = [jaxley>=0.13]
viz = [matplotlib, plotly, dash]
tutorials = [jupyter, nbformat, nbconvert, ipykernel, matplotlib]
dev = [pytest, pytest-cov, ruff, black]
all = [jax, jaxfne, jaxley, viz, tutorials, dev]
```

Guard pattern for imports:
```python
try:
    import jaxfne as jtfne
except ImportError:
    raise ImportError("Install with: pip install -e '.[jaxfne]'")
```

Tests skip cleanly when optional dependencies absent via `pytestmark = pytest.mark.skipif(...)`.

---

## v0.0.3: Canonical Imports

✅ **COMPLETE**

- `src/jbiophysic/jaxfne_integration.py`: Guarded jaxfne import with clear error message
- `src/jbiophysic/jaxfne_advanced.py`: Same guard pattern
- `tests/test_jaxfne_integration.py`: Skips when jaxfne absent
- `tests/test_jaxfne_advanced.py`: Skips when jaxfne absent

---

## v0.0.4: Optional Guard Discipline

✅ **COMPLETE**

- `tests/test_import_smoke.py`: Now conditionally checks optax symbols
- Smoke test passes without [jax] or [jaxfne] extras
- Core imports (version, tfne, optim.Bound) always available
- Optax-backed symbols checked only if optax installed

---

## v0.0.5: README Cleanup

✅ **COMPLETE**

- Removed stale test counts ("106 passed, 8 skipped")
- Added atlas role clarification
- Updated dependency descriptions
- Added jaxfne extra in install instructions
- Documented atlas role: teaches biophysics, uses jaxfne as backend when needed

---

## v0.0.6: Namespace Clarity

✅ **DOCUMENTED**

- `jbiophysic.jtfne` is the spectrolaminar workflow API (not a jaxfne wrapper)
- Added deprecation notice: future rename to jbiophysic.atlas or jbiophysic.workflows
- Current code continues to work (backward compatible)
- Rename deferred to v0.1 or later (non-breaking change when ready)

---

## v0.0.7–v0.0.11: Standards (To Document)

### v0.0.7: Tutorial Standard
Every chapter has:
- Concept map and doctrine
- Mathematical glossary
- Minimal executable model
- Diagnostics and validation
- Visualization and figures
- Parameter sweep or null control
- jaxfne bridge (if source-field relevant)
- Notebook tutorial with exercises
- Manifest and claim gates

### v0.0.8: Smoke Test Standard
- Core imports without extras
- Version accessible and correct
- Optional symbols guarded (gsdr_direction if optax)
- No bare ImportError on missing optional deps

### v0.0.9: Claim-Language Audit
All documentation, docstrings, and figures mark:
- truth_safe_unverified
- Computational/pedagogical vs. biological
- Assumption limits
- Solver status (proxy/simplified/unsolved)

### v0.0.10: CI Cleanup
- Tests pass without optional extras (or skip cleanly)
- Optional dependency matrix in CI
- Smoke path works on minimal install
- No hardcoded test counts in README/CI

### v0.0.11: Chapter 0 Release Report
- All v0.0.0–v0.0.10 complete
- Audit trail: commits, PRs, testing
- Ready for v0.1 (units & dimensions)

---

## Key Design Constraints

### For every future feature, ask:

1. **Is this a reusable source-field/readout engine contract?**
   → Put it in `jaxfne`

2. **Is this a biophysical concept, case study, teaching chapter, diagnostic, or explanatory notebook?**
   → Put it in `jbiophysic`

---

## Next: v0.1.0 (Units, Dimensions, Numerical Discipline)

Goal: Make biophysical quantities explicit before equations become complex.

Chapters:
- v0.1.0: Units doctrine
- v0.1.1: Dimensional glossary
- v0.1.2: Unit conversion helpers
- v0.1.3: Float32/float64 comparison
- v0.1.4: Stability diagnostics
- v0.1.5: Time-step sweep
- v0.1.6: Null test
- v0.1.7: jaxfne manifest alignment
- v0.1.8: Notebook
- v0.1.9: Exercises
- v0.1.10: Tests
- v0.1.11: Release

---

**Status:** v0.0 package hygiene complete. Ready for v0.1 (units & numerics).
