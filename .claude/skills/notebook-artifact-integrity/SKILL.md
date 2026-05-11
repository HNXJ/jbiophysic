# Notebook Artifact Integrity

## Purpose

Prevent stale, mislabeled, or corrupted notebook artifacts. Validate source parsing, execution counts, output freshness. Distinguish portable Jupyter notebooks (nbconvert-executable) from Colab artifacts (google.colab + magics). Detect newline-collapsed JSON source.

## When to use

- Before committing notebook changes
- Before moving or renaming notebooks
- When updating tutorials (00-04 or source notebooks)
- When archiving Colab artifacts
- Before creating new portable tutorials
- When notebooks fail to parse or execute

## Inputs to inspect first

- `docs/tutorial_status.md` (notebook classification: Portable, Colab, Source/Reference)
- `README.md` tutorials section (lists portable and Colab)
- `.ipynb` files in `tutorials/` directory
- Evidence of changes in notebook source cells (newline collapse, missing imports)

## Standard commands

```bash
# Locate all notebooks
find tutorials -name "*.ipynb" -type f

# Check for Colab magic commands and google.colab imports
grep -l "google.colab\|%cd\|!pip\|!shell" tutorials/**/*.ipynb 2>/dev/null || echo "No Colab artifacts detected"

# Parse and validate notebook JSON (Python snippet below)
python - <<'PY'
import ast, json, pathlib
notebook_path = pathlib.Path("tutorials/NOTEBOOK.ipynb")
nb = json.loads(notebook_path.read_text())
for i, cell in enumerate(nb["cells"]):
    if cell.get("cell_type") == "code":
        src = "".join(cell.get("source", [])) if isinstance(cell.get("source", []), list) else cell.get("source", "")
        try:
            ast.parse(src)
        except SyntaxError as e:
            print(f"Cell {i} syntax error: {e}")
    if cell.get("cell_type") == "code" and "execution_count" in cell:
        if cell["execution_count"] is None or cell["execution_count"] < 1:
            print(f"Cell {i} missing execution count")
print("AST parse OK")
PY

# Execute notebook via nbconvert
jupyter nbconvert --to notebook --execute tutorials/NOTEBOOK.ipynb --output tutorials/NOTEBOOK.executed.ipynb

# Check output freshness
sha256sum tutorials/NOTEBOOK*.ipynb

# Compare execution counts
jupyter nbconvert --to python tutorials/NOTEBOOK.ipynb --stdout | grep "# In\[" | wc -l
```

## Safe procedure

1. Identify notebook type (Portable, Colab, or Source)
2. If Portable:
   - Verify no google.colab, %cd, !pip, !shell magics
   - Parse JSON and validate AST for all code cells
   - Execute with nbconvert: `jupyter nbconvert --to notebook --execute NOTEBOOK.ipynb --inplace`
   - Verify execution counts are sequential (1, 2, 3, ...)
   - Verify outputs are present and recent
   - Calculate SHA256 for audit trail
3. If Colab:
   - Verify it's labeled with `.colab.ipynb` suffix
   - Document evidence (google.colab import, magic commands)
   - Archive in `tutorials/source_notebooks/` or separate archive dir
   - Do NOT execute in standard Jupyter; mark "Reference only"
4. If Source/Reference:
   - Mark clearly as "WIP" or "reference"
   - Note if outputs are stale or incomplete
5. Commit only updated notebooks; verify git diff shows realistic changes

## Validation gate

```bash
source .venv/bin/activate
# Parse all portable notebooks
find tutorials -name "*.ipynb" ! -name "*.colab.ipynb" ! -name "*source*" -exec python - <<'PY'
import ast, json, pathlib, sys
p = pathlib.Path(sys.argv[1])
nb = json.loads(p.read_text())
for i, cell in enumerate(nb["cells"]):
    if cell.get("cell_type") == "code":
        src = "".join(cell.get("source", []))
        try:
            ast.parse(src)
        except SyntaxError as e:
            print(f"FAIL: {p} cell {i}: {e}")
            sys.exit(1)
PY
{} \;

# Optionally execute a portable notebook
jupyter nbconvert --to notebook --execute tutorials/00_neuronal_equations_book.ipynb --inplace

# Verify test still passes
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=src python -m pytest -q --tb=short
```

## Stop conditions

- If AST parse fails (syntax error in notebook cell source)
- If execution_count is missing or not sequential
- If nbconvert execution hangs or exits with error
- If outputs are missing in a supposedly executed notebook
- If a Colab artifact is mixed with portable tutorials (should be archived separately)
- If newline collapse is detected (source cells are concatenated without newlines)

## Final report fields

- Notebooks audited: (list and count)
- Portable notebooks: (count, all passing)
- Colab artifacts: (count, labeled, archived status)
- Source/reference notebooks: (count, status)
- AST parse errors: (0 or list of failures)
- Execution count validation: (pass/fail)
- nbconvert execution: (success or failure)
- Output freshness: (recent/stale)
- SHA256 checksums: (for audit trail)
- Recommendation: (safe to commit / revise / blocked)
