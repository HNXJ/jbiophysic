# Skill: TFNE Source-Field-Probe Contract

**Purpose:** Operational skill for TFNE source projection, fields, CSD, and probe/readout work.

**Use when:** Editing:
- `src/jbiophysic/tfne/fields.py`
- `src/jbiophysic/tfne/sources.py`
- `src/jbiophysic/tfne/tensors.py`
- `src/jbiophysic/tfne/csd.py`
- `src/jbiophysic/tfne/solvers.py`
- future `src/jbiophysic/tfne/probes.py`

## The Contract
- **Normalized Kernels:** Source kernels must be normalized (integral = 1).
- **Current Conservation:** Integrated current must be conserved (Emitter -> Source).
- **Gauge Fixing:** Gauge must be fixed (e.g., mean-zero or pinned) before interpreting `phi_e`.
- **SPD Tensors:** Passive conductivity/admittivity tensors must be Symmetric Positive Definite (SPD).
- **CSD Convention:** CSD sign convention must be explicitly declared.
- **Solver Status:** Solver residual and convergence status must be reported.
- **Probe Definition:** Probe units and geometry must be explicitly declared.

## Required Tests
- Kernel integral equals 1.
- Source projection conserves current.
- Mean-zero or pinned gauge works.
- Constant vector field divergence is zero.
- Finite `phi_e`, `J_e`, and CSD values.
- SPD tensor has positive eigenvalues.

## Acceptance Gate
- No field interpretation without source conservation, gauge, and finite-value checks.
