# Skill: TFNE Release Candidate (RC) Audit

**Purpose:** Avoid repeating release, CI, or reference confusion.

**Use when:** Before a release tag or after substantial TFNE/notebook changes.

## Required Checks
- `git status --short`
- `git fetch origin --prune --tags`
- Local `main` equals `origin/main`.
- Latest GitHub Actions run for current HEAD is "success".
- Version metadata matches intended release version.
- Required files are tracked (e.g., `git ls-files src/jbiophysic/data tests/data`).
- `.gitignore` does not accidentally hide source packages or tests.

## Artifact Archive Rule
- Release ZIP must be generated from the tag, not from a stale `main.zip`.

## Stop Conditions
- Dirty tree.
- Stale release archive.
- Local-only files intended for release.
- Untracked tests.
- CI pending or failing.
