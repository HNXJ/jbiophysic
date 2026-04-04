# jbiophysics
> Advanced Biophysical Modeling for Omission Prediction (Jaxley/JAX)

## Highlights
- **Fluent API**: `NetBuilder` and `OptimizerFacade` for high-level simulation.
- **Biophysical Primitives**: `SafeHH`, `Inoise`, and graded synapses.
- **Cortical Logic**: Two-column V1 + HO architecture for omission modeling.
- **Optimization**: SDR, GSDR, and AGSDR tuning for synaptic conductances.
- **Reporting**: Automated Plotly dashboards, Reveal.js slides, and base64 API.

## Project Structure
- `jbiophysics/core`: Mechanisms (HH, Synapses) and Optimizers.
- `jbiophysics/systems`: Pre-built network architectures (V1, Omission).
- `jbiophysics/viz`: Visualization utilities (Spectrograms, Rasters).
- `jbiophysics/scripts`: Production runners (simulation, tuning, reporting).
- `api.py`: FastAPI backend on port 7701.

## Environment & Scripts
- Recommended Python: 3.11 (`.venv_311/bin/python`).
- `config.md`: Project settings (see `config.template.md`).
- Primary entry point: `jbiophysics/scripts/run_omission_trial.py`.

## Audit Results (2026-04-04)
A full refactor of the repository was completed to resolve 32 severe issues, including:
- Missing `systems/` and `viz/` directory recreation.
- `SafeHH(name="HH")` naming fix across all builders.
- Duplication and typo fixes in GSDR/AGSDR modules.
- Absolute paths removed; environment logic improved.
- Syntax and import bug fixes in runners and API.
- Re-implementation of 5+ missing viz/network modules.

---
*Created and maintained by Antigravity.*
