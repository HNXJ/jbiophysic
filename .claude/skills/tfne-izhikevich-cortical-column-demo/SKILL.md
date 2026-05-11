# Skill: TFNE-Izhikevich Cortical Column Demo

**Purpose:** Guide creation or modification of cortical-column TFNE-Izhikevich examples and notebooks.

**Use when:** Creating examples/tutorials like:
- V1/PFC two-column Izhikevich notebooks.
- Laminar E/PV/SST/VIP anatomy demos.
- 1 mm depth, 0.1 mm radius cortical tubes.
- Plotly 3D anatomy visualizations.
- Spontaneous baseline smoke simulations.

## Required Demo Gates
- **Colab Compatibility:** Import repo in Colab via clone/install or editable local install.
- **Determinism:** Use a deterministic seed.
- **SI Units:** Declare geometry in SI units (e.g., meters).
- **Coordinate Hygiene:** No overlapping neuron coordinates below declared threshold.
- **Smoke Assertion:** Nonzero spiking smoke assertion.
- **Stats Reporting:** Report per-area and per-cell-type spike counts.
- **Artifact Hygiene:** Generated HTML/output artifacts not committed unless explicitly requested.
- **Truth Status:** Explicitly state exploratory/computational status only.

## Visual Conventions (Plotly 3D)
- **E:** gold/yellow
- **PV:** cyan
- **SST:** magenta
- **VIP:** white/gray
- **Context:** Column shell/wireframe must be visible.
- **Interactivity:** Hover shows neuron_id, area, layer, cell_type, coordinates, and spike count.

## Validation
- Execute notebook or run notebook smoke via nbconvert.
- Run targeted visualization tests if Plotly is installed.
- Run full test suite if any source code was changed.
