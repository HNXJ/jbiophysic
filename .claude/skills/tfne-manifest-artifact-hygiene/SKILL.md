# Skill: TFNE Manifest and Artifact Hygiene

**Purpose:** Keep simulation/tutorial artifacts reproducible without polluting git.

**Use when:** Examples produce `.npz`, `.csv`, `.json`, `.html`, notebook outputs, field snapshots, spike matrices, or hashes.

## Rules
- **Output Location:** Generated outputs go under `outputs/` or a user-specified artifact directory.
- **Git Hygiene:** Do NOT commit generated outputs by default.
- **Fixture Policy:** Commit small fixtures only if they are source/test fixtures.
- **Manifest Content:** Manifests must include: code version, seed, geometry, units, calibration status, output paths, hashes, and truth mode.

## Required Checks
- `git status --short --ignored`
- Verify no `.mat`, huge outputs, secrets, or private paths.
- Hash manifest if generated artifacts are part of a report.

## Stop Condition
- Generated data staged or committed unintentionally.
