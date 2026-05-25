# jaxfne Usage Refactor Inventory

**Date:** 2026-05-24  
**Status:** Stage 1 — Code Classification  
**Goal:** Map jbiophysic modules for jaxfne-first refactoring

---

## Module Classification Summary

| Module | Classification | Current Status | Action | Notes |
|--------|-----------------|-----------------|--------|-------|
| `__init__.py` | atlas_keep | public API exports | review exports | May need to update if jtfne → workflows |
| `_optional.py` | atlas_keep | optional import guard | keep as-is | Likely supports optional jaxfne |
| `passive_membrane/` | atlas_keep | pedagogical | keep native | v0.2 teaching chapter |
| `hodgkin_huxley/` | atlas_keep | pedagogical | keep native | v0.3 teaching chapter |
| `cells/izhikevich.py` | atlas_keep | cell model | keep native | Teaching model, may interface jaxfne |
| `cells/hh.py` | atlas_keep | cell model | keep native | Teaching model |
| `circuits/` | atlas_keep | network scaffold | keep native | Learning architecture |
| `analysis/` | atlas_keep | metrics/diagnostics | expand for jaxfne outputs | Extend with jaxfne-compatible diagnostics |
| `jtfne.py` | workflow_keep (legacy monolith) | spectrolaminar workflow | **MOVE → workflows/spectrolaminar** | Deprecate old import; make it workflow layer |
| `jaxfne_integration.py` | jaxfne_bridge | integration shim | **REFACTOR → bridges/jaxfne** | Move to canonical bridge namespace |
| `jaxfne_advanced.py` | jaxfne_bridge or legacy_engine_duplicate | advanced features | **AUDIT & MOVE** | Clarify if engine duplication or bridge |
| `tfne/sources.py` | legacy_engine_duplicate | TFNE source projection | **DEPRECATE / MOVE TO LEGACY** | jaxfne has source contracts; do not duplicate |
| `tfne/fields.py` | legacy_engine_duplicate | field/proxy solver | **DEPRECATE / MOVE TO LEGACY** | jaxfne has field contracts; do not duplicate |
| `tfne/csd.py` | legacy_engine_duplicate | CSD readout | **DEPRECATE / MOVE TO LEGACY** | jaxfne has readout contracts; do not duplicate |
| `tfne/solvers.py` | legacy_engine_duplicate | field solver | **DEPRECATE / MOVE TO LEGACY** | jaxfne has solver contracts; do not duplicate |
| `tfne/tensors.py` | atlas_keep | data structure | keep if not duplicating jaxfne | Audit tensor layouts vs jaxfne |
| `tfne/operator_status.py` | workflow_keep | operator metadata | keep & integrate with jaxfne bridge | Extend for jaxfne-backed runs |
| `conditions.py` | atlas_keep | experimental conditions | keep native | Configuration/scenario definition |
| `objectives/` | workflow_keep or legacy_engine_duplicate | optimization objectives | **AUDIT & MOVE** | If using jaxfne optimizer, delegate; else refactor |

---

## Detailed Module Inspection Notes

### Teaching Modules (atlas_keep) — **NO CHANGES REQUIRED**

- `passive_membrane/` — v0.2 pedagogical chapter; native jbiophysic implementation
- `hodgkin_huxley/` — v0.3 pedagogical chapter; native jbiophysic implementation (recently added)
- `cells/izhikevich.py` — Izhikevich cell model; teaching module; may be used as source in jaxfne bridge
- `cells/hh.py` — Hodgkin-Huxley cell model; teaching module
- `circuits/` — Network construction and topology; teaching architecture

**Decision:** Keep these modules native. They teach biophysics. jaxfne workflows may use these as inputs (e.g., Izhikevich cells as sources).

---

### Workflow Modules (workflow_keep + refactoring) — **MOVE & DEPRECATE OLD IMPORT**

#### `jtfne.py` — **MAJOR REFACTORING REQUIRED**

**Current Status:**
- Large (~500+ lines) spectrolaminar workflow monolith
- Contains config, construction, simulation, evaluation, optimization logic
- Docstring explicitly states it should be renamed to `workflows` or `atlas`
- **Not a wrapper around jaxfne** — it is an orchestrator/workflow layer

**Problem:**
- Naming confusion: `jtfne` implies `jaxfne` but is actually workflow code
- Blocks canonical naming of `import jaxfne as jtfne`
- Large monolith mixes scenario definition with execution

**Action:**
1. Move core workflow logic to:
   ```
   src/jbiophysic/workflows/spectrolaminar/
     __init__.py
     config.py
     build.py
     run.py
     evaluate.py
     figures.py
     reports.py
   ```
2. Leave deprecation shim at `src/jbiophysic/jtfne.py`:
   ```python
   import warnings
   warnings.warn(
       "jbiophysic.jtfne is deprecated. Use jbiophysic.workflows.spectrolaminar "
       "or import jaxfne as jtfne for the jaxfne engine.",
       DeprecationWarning,
       stacklevel=2,
   )
   from jbiophysic.workflows.spectrolaminar import *
   ```

---

### Bridge Modules (jaxfne_bridge) — **CREATE CANONICAL NAMESPACE**

#### `jaxfne_integration.py` — **MOVE TO bridges/jaxfne/convert.py**

**Current Status:** Integration shim connecting jbiophysic configs to jaxfne calls

**Action:**
- Move to `src/jbiophysic/bridges/jaxfne/convert.py`
- Create canonical namespace `jbiophysic.bridges.jaxfne`

#### `jaxfne_advanced.py` — **AUDIT & INTEGRATE**

**Current Status:** Advanced features (unclear if bridge or engine duplicate)

**Action:** Inspect file to determine if:
- Bridge adapter → move to `bridges/jaxfne/advanced.py`
- Engine duplicate → move to `legacy/`
- Mixed → split accordingly

---

### Legacy/Duplicate Engine Modules (legacy_engine_duplicate) — **DEPRECATE OR MOVE**

#### `tfne/sources.py`, `tfne/fields.py`, `tfne/csd.py`, `tfne/solvers.py`

**Problem:** These modules likely duplicate jaxfne engine contracts:
- jaxfne owns source projection
- jaxfne owns field/proxy solvers
- jaxfne owns readout (LFP/CSD/EEG/MEG) contracts
- jbiophysic should not maintain parallel implementations

**Action:**
1. Inspect each module to confirm duplication
2. If duplicates jaxfne contracts:
   - Move to `src/jbiophysic/legacy/tfne/` 
   - Mark with `status = "not_canonical"` or `"reference_only"`
   - Update docstring: "This is a reference/teaching implementation. For production use, dispatch to jaxfne."
3. If bona fide jbiophysic-only logic:
   - Keep but audit for consistency with jaxfne semantics
4. Update imports in jtfne/workflows to use jaxfne versions

---

### Analysis Modules (atlas_keep + extend) — **EXTEND FOR jaxfne OUTPUTS**

#### `analysis/` — Keep native + add jaxfne adapters

**Current Status:** Metrics, diagnostics, fano, spectra, spikes, synchrony

**Action:**
- Keep native implementations for pedagogical use
- Add jaxfne-compatible diagnostic adapters:
  ```
  analysis/jaxfne_adapters.py
    - convert jaxfne operator_status to jbiophysic metrics
    - convert jaxfne field readouts to jbiophysic diagnostics
    - unify CSD/LFP/EEG/MEG readout formats
  ```

---

## Directory Structure After Refactoring

```
src/jbiophysic/
├── __init__.py                          [review exports]
├── _optional.py                         [keep]
├── conditions.py                        [keep]
├── passive_membrane/                    [keep]
├── hodgkin_huxley/                      [keep]
├── cells/                               [keep]
├── circuits/                            [keep]
├── analysis/                            [keep + extend]
│   └── jaxfne_adapters.py               [NEW]
├── bridges/                             [NEW CANONICAL NAMESPACE]
│   ├── __init__.py
│   └── jaxfne/
│       ├── __init__.py                  [public API]
│       ├── config.py
│       ├── convert.py                   [from jaxfne_integration.py]
│       ├── run.py                       [orchestration]
│       ├── reports.py                   [manifests & validation]
│       └── validation.py
├── workflows/                           [NEW WORKFLOW LAYER]
│   ├── __init__.py
│   └── spectrolaminar/
│       ├── __init__.py                  [public API]
│       ├── config.py                    [from jtfne.py config section]
│       ├── build.py                     [from jtfne.py construct section]
│       ├── run.py                       [from jtfne.py simulate/optimize]
│       ├── evaluate.py                  [from jtfne.py evaluate]
│       ├── figures.py                   [from jtfne.py figure generation]
│       └── reports.py                   [from jtfne.py manifest section]
├── legacy/                              [NEW ARCHIVE FOR DEPRECATED CODE]
│   ├── tfne/                            [old TFNE solver code]
│   │   ├── sources.py
│   │   ├── fields.py
│   │   ├── csd.py
│   │   └── solvers.py
│   └── jtfne_deprecated.py              [old monolith marker]
├── jtfne.py                             [DEPRECATION SHIM ONLY]
└── tfne/                                [remove or minimal]
    └── operator_status.py               [keep, integrate with bridges]
```

---

## Classification Decision Rules

**atlas_keep:**
- Teaching/pedagogical content
- Foundational cell models (Izh, HH)
- Network topology/construction (no engine duplication)
- Configuration/scenario definition
- Analysis metrics

**workflow_keep:**
- Scenario-specific orchestration (spectrolaminar, omission, etc.)
- Configuration templating
- Evaluation pipelines
- Figure generation

**jaxfne_bridge:**
- Adapters from jbiophysic configs to jaxfne calls
- Conversion/normalization layers
- Manifest generation for jaxfne runs
- Validation gates for jaxfne outputs

**legacy_engine_duplicate:**
- Old TFNE solvers / source projections / field readouts
- Parallel implementations of jaxfne-owned contracts
- Reference implementations (not canonical)

**deprecated_shim:**
- Old import paths (e.g., `from jbiophysic import jtfne`)
- Backward-compatibility wrappers

---

## Next Steps (Stage 2+)

1. **Stage 2:** Create `src/jbiophysic/bridges/jaxfne/` with canonical API
2. **Stage 3:** Move `jtfne.py` logic to `workflows/spectrolaminar/` + deprecation shim
3. **Stage 4:** Audit and move/deprecate `tfne/` duplicate modules
4. **Stage 5:** Extend `analysis/` with jaxfne adapters
5. **Stage 6:** Create figure-suite script using jaxfne bridge + workflows

---

## Summary

- **Teaching modules:** Keep native (atlas_keep)
- **Workflow modules:** Reorganize into `workflows/` hierarchy (workflow_keep + refactor)
- **Bridge modules:** Consolidate into `bridges/jaxfne/` (jaxfne_bridge)
- **Engine duplicates:** Move to `legacy/` (legacy_engine_duplicate)
- **Deprecation:** Create shim for old `from jbiophysic import jtfne`

**Goal:** jbiophysic becomes a jaxfne-first biophysical atlas. Workflows dispatch to jaxfne engine. Teaching modules remain native.
