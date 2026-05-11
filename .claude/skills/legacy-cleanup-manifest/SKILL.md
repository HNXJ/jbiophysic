# Legacy Code Cleanup Manifest

## Purpose

Remove legacy and unused code safely. No deletion without a manifest documenting grep/import/reference audit, test coverage impact, and rollback strategy. Each deletion candidate must be audited before removal.

## When to use

- Before deleting any module, class, function, or file
- When cleaning up dead code paths
- When removing compatibility shims or experimental code
- When archiving modules marked as legacy
- Before deleting test files or fixtures

## Inputs to inspect first

- `scratch/` directory (legacy experimental code)
- Files marked with `# deprecated`, `# legacy`, `# WIP`, `# TODO delete`
- Modules with zero test coverage
- Imports in `__init__.py` that may reference removed code
- `CLAUDE.md` Section K (Legacy cleanup policy: deletion by manifest only)
- `P1_DECISIONS.md` (Legacy policy: manifest, grep/import audit, no silent deletion)

## Standard commands

```bash
# Find likely legacy files
find src scratch -type f -name "*.py" | xargs grep -l "deprecated\|legacy\|WIP\|TODO.*delete" | sort

# For each candidate, audit its references
CANDIDATE="src/jbiophysic/path/to/module.py"
SYMBOL="function_or_class_name"

# Grep for all references
grep -RIn "$SYMBOL" src tests docs README.md pyproject.toml

# Check imports
grep -RIn "from.*import.*$SYMBOL\|import.*$SYMBOL" src tests

# Check test coverage (if any)
grep -RIn "test.*$SYMBOL\|$SYMBOL.*test" tests

# Before deleting, verify no other references exist
git grep "$SYMBOL" || echo "No references in git"
```

## Safe procedure

**For each legacy code candidate:**

1. **Create deletion manifest** (as a comment block or separate doc):
   ```
   DELETION MANIFEST
   ─────────────────
   Candidate: <path/to/file.py or function_name>
   Reason: <orphaned / unused / superseded by X / experimental>
   
   Grep results:
   - Symbol references: <count or none>
   - Import statements: <count or none>
   - Test references: <count or none>
   - Files affected: <list or none>
   
   Archive decision: <delete / move to scratch/archive/>
   Rollback: <commit SHAto restore if needed>
   ```

2. **Perform full audit:**
   - `grep -RIn "<symbol>"` in src/, tests/, docs/
   - `grep -RIn "import <symbol>"` everywhere
   - Check `__init__.py` for public exports
   - Check if any test files depend on this module

3. **Determine scope:**
   - Is it safe to delete (no references)?
   - Or should it be moved to `scratch/legacy_archive/`?
   - Are there any tests that will break?

4. **If deleting:**
   - Verify no references exist
   - Run full test suite before commit
   - Include manifest comment in commit message
   - Stage only the deleted file (no other changes)

5. **If archiving:**
   - Move to `scratch/legacy_archive/<name>/`
   - Include README explaining what it is and why it was archived
   - Stage the archive, remove from src
   - Run tests

6. **Do NOT delete without:**
   - Completed manifest
   - Full grep audit with results in commit message
   - Test suite passing after deletion
   - Explicit user confirmation (manifest + results)

## Validation gate

```bash
# Before deletion commit, verify:
source .venv/bin/activate
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=src python -m pytest -q --tb=short
# Expected: 61 passed, 0 failed (same as before deletion)

# Check git diff is clean (only deletions, no accidental changes)
git status --short
git diff --stat  # Should show "X files changed, Y insertions(-), Z deletions(-)"

# Verify no remaining references to deleted code
grep -RIn "<deleted-symbol>" src tests || echo "✓ No lingering references"
```

## Manifest template

```
═══════════════════════════════════════════════
LEGACY DELETION MANIFEST
═══════════════════════════════════════════════

CANDIDATE: src/jbiophysic/path/to/legacy_module.py

REASON FOR DELETION:
- [orphaned after refactor]
- [superseded by new_module.py]
- [experimental, not used]
- [test coverage: 0%]

AUDIT RESULTS:
─────────────
Grep references in src/:
  - None

Grep references in tests/:
  - None

Grep references in docs/:
  - None

Import statements:
  - None

Files importing this module:
  - None

Dependent tests:
  - None

Public export in __init__.py:
  - No

═══════════════════════════════════════════════

DECISION: DELETE (safe, no dependencies)
ARCHIVE DECISION: Delete (not referenced elsewhere)

ROLLBACK:
- Commit SHAat point of deletion: <SHA>
- Recovery: git show <SHA>:src/jbiophysic/path/to/legacy_module.py > recovery.py

═══════════════════════════════════════════════
```

## Stop conditions

- If any references are found in src/, tests/, or docs/
- If test coverage changes after deletion (tests still pass but fewer tested paths)
- If imports fail after deletion
- If __init__.py exports the deleted symbol
- If deletion is attempted without completed manifest
- If any test fails after deletion

## Final report fields

- Candidates identified: (list and count)
- Audit completed: (yes/no, all candidates)
- Candidates deleted: (list and count)
- Candidates archived: (list and count)
- Deletion manifests provided: (yes/no)
- Grep references found: (0 or list)
- Test baseline before: (61 passed)
- Test baseline after: (pass/fail)
- Rollback method documented: (yes/no)
- Recommendation: (all safe / revise / blocked)
