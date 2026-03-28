---
name: analysis-jbiophysics-api-dashboard
description: FastAPI backend and interactive visualization dashboard for the jbiophysics repository.
---
# analysis-jbiophysics-api-dashboard

This skill covers the computational backend and interactive visualization suite for the `jbiophysics` repository. It allows for remote simulation control and real-time monitoring of biophysical experiments.

## 1. FastAPI Backend (Port 7701)
The `jbiophysics.api` module exposes the modeling engine via REST endpoints.

### Key Endpoints
- **POST `/simulate/v1`**: Executes a 200-neuron V1 column simulation.
- **POST `/simulate/omission`**: Executes a 5000ms two-column omission trial.
- **POST `/tuning/run`**: Starts a background GSDR tuning session.
- **GET `/tuning/status`**: Returns real-time epoch and loss metrics.
- **GET `/visualize`**: Returns a JSON payload containing Base64-encoded PNGs (Raster, LFP, TFR).

## 2. Visualization Suite (`jbiophysics/viz/`)
- **Interactive Dashboards**: Powered by Plotly for local browser-based analysis.
- **Static Reports**: Matplotlib-based `omission_viz.py` optimized for API streaming.
    - **Raster Plots**: Coloured by cell type (L23, L4, L56, PV, SST, VIP).
    - **LFP Traces**: Dual-column comparisons with Gaussian smoothing.
    - **TFR Spectrograms**: Morlet wavelet transforms highlighting power in canonical bands.

## 3. Agent Bridge
The `/agent/ask` endpoint allows for interactive network design by relaying prompts to a local LLM (e.g., Qwen-3.5-9B). The agent provides JSON-formatted `NetworkConfig` blocks which can be directly fed into the `/build` endpoint to instantiate new architectures.

## 4. Usage Commands
```bash
# Start the backend
uvicorn jbiophysics.api:app --port 7701
# Trigger an omission trial
curl -X POST http://localhost:7701/simulate/omission -d '{"bu_on": false, "td_on": true}'
```
