---
name: analysis-jbiophysics-api-dashboard
description: Scientific plotting, dashboard management, and API orchestration for jbiophysics.
---
# analysis-jbiophysics-api-dashboard

This skill documents the visualization and analysis layer for hierarchical biophysical simulations.

## 1. Scientific Plotting Protocols
- **Raster Plots**: Always order by `Area.Population` (e.g., Sensory at the bottom, Executive at the top) to visualize temporal hierarchical flow.
- **TFR Analysis**: Use the manual **Morlet Wavelet** implementation for Time-Frequency Representations; `scipy.signal.cwt` is deprecated and often incompatible with JAX traces.
- **Spectral Matching (SSS)**: Use log-scale Power Spectral Density (PSD) for comparing simulations with empirical monkey data. Focus on peak detection (Alpha/Beta peaks).

## 2. API & Real-Time Monitoring
- **Backend Persistence**: The JBiophysics API typically persists on **Port 7701**.
- **Live Tuning**: Monitor optimization trajectories via `/tuning/status`. Gradients are streamed as JSON for real-time visualization of the 3D error surface.
- **Interactive Dashboard**: Use Plotly for interactive exploration of traces and rasters. Ensure `MPLCONFIGDIR=/tmp/matplotlib_cache` is set to avoid font-cache permission issues on remote compute.

## 3. Highly Useful Hints for Analysis
- **LFP Calculation**: Use the `LFP_tools` in `systems/visualizers/` to compute virtual Local Field Potentials from the somatic currents of pyramidal populations.
- **Kappa Targeting**: Use `compute_kappa()` in `viz/` to confirm the model is in the asynchronous-irregular regime (Kappa < 0.1).
- **Report Generation**: Use `scripts/build_html_report.py` to consolidate simulation artifacts into a single sharable dashboard.
- **Export Formats**: Prefer `.json` for metadata and `.pkl` for optimized parameter sets. Use `report.export_json()` to capture the full state of an `OptimizerFacade` run.

## 4. Common Pitfalls
- **Sampling Artifacts**: Ensure `delta_t` matches the recording resolution (typically 40 kHz or 0.025ms). Incorrect resolution will lead to spectral aliasing.
- **VRAM Exhaustion**: Large-scale TFR analysis can consume high VRAM on M3 Max. Perform TFR sequentially or on downsampled traces for large hierarchies.
- **Color Palettes**: Adhere to the project style: **Madelane Golden Dark** (#CFB87C Gold / #9400D3 Violet) for publication-quality figures.
