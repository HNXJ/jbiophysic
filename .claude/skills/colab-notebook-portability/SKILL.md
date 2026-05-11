# Skill: Colab Notebook Portability

**Purpose:** Make tutorials portable between local repo and Google Colab.

**Use when:** Creating or editing notebooks under `tutorials/` or release notebooks.

## Required First-Cell Pattern
- Detect Colab environment.
- Clone `https://github.com/HNXJ/jbiophysic.git` if needed.
- Optionally checkout `v1.0.1` or specified tag.
- Install needed extras: `pip install ".[jax,viz,tutorials]"`.
- Add `src` to `sys.path`.
- Print `jbiophysic.__file__` for verification.

## Constraints
- **No Hardcoded Paths:** Avoid hardcoded local paths.
- **No Drive Dependency:** Must not rely on mounted Google Drive unless explicitly documented.
- **Small Artifacts:** Keep generated artifacts small or use explicit external storage.

## Validation
- Notebook JSON parses correctly.
- First import cell can run in a clean environment.
- Notebook has a title and a "claim-status" markdown cell (e.g., Exploratory).
- Outputs are fresh if committed.
