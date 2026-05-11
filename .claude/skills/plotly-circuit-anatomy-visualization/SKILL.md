# Skill: Plotly Circuit Anatomy Visualization

**Purpose:** Operational skill for Plotly circuit visualization and `jbiophysic.viz.network3d`.

**Use when:** Editing:
- `src/jbiophysic/viz/network3d.py`
- `examples/plot_laminar_circuit_3d.py`
- Notebooks with Plotly anatomy views.

## Required Behavior
- **3D Rendering:** Accept 1D/2D/3D coordinates but always render in 3D.
- **Coordinate Preservation:** Preserve coordinates unless deterministic duplicate-jitter is requested.
- **Contextual View:** Show layer and column context.
- **Pathway Traces:** Support traces for `thalamic_input`, `feedforward`, `feedback`.
- **Trace Separation:** Separate traces by cell type for legend clarity.
- **Metadata:** Hover metadata includes neuron_id, area, layer, cell_type, x/y/z.

## Required Tests
- Dictionary input support.
- Population-object input support.
- 1D/2D/3D coordinate handling.
- Deterministic duplicate jitter.
- HTML write smoke test.
- No all-to-all default clutter.

## Stop Condition
- If Plotly is absent and the task requires rendering, install/use `[viz]` extra or report BLOCKED.
