"""
jbiophysics FastAPI Backend — Omission Paradigm Edition.

Port: localhost:7701

Endpoints:
  POST /build                  — generic NetBuilder (original)
  POST /simulate/{net_id}      — generic simulation (original)
  POST /simulate/v1            — 200-neuron V1 baseline
  POST /simulate/omission      — 5000ms two-column omission trial
  GET  /tuning/status          — live GSDR epoch + loss
  GET  /tuning/metrics         — RSA, Kappa, SSS per epoch history
  GET  /visualize              — Base64 PNG raster+LFP+TFR
  POST /agent/ask              — local model relay (original)
"""

import os
import sys
import uuid
import json
import threading
import numpy as np
from typing import Dict, List, Optional, Any
from fastapi import FastAPI, HTTPException, Body
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

# ── Path bootstrap ─────────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from jbiophysics.compose import NetBuilder
from jbiophysics.export import ResultsReport
from jbiophysics.viz.dashboard import generate_dashboard
from jbiophysics.viz.omission_viz import plot_omission_raster, plot_lfp_traces, plot_tfr

# ── Pydantic models ────────────────────────────────────────────────────────────

class PopulationConfig(BaseModel):
    name: str
    n: int
    cell_type: str = "pyramidal"
    noise_amp: float = 0.05
    noise_tau: float = 20.0

class ConnectionConfig(BaseModel):
    pre: str
    post: str
    synapse: str
    p: float = 0.1
    g: Optional[float] = None

class NetworkConfig(BaseModel):
    seed: int = 42
    dt: float = 0.1
    t_max: float = 1000.0
    populations: List[PopulationConfig]
    connections: List[ConnectionConfig]
    trainable: List[str] = []

class V1SimConfig(BaseModel):
    seed:  int   = 42
    dt:    float = 0.025
    t_max: float = 1000.0
    stim_times_ms: List[float] = [200.0, 500.0, 800.0]
    stim_amp:      float = 2.0

class OmissionSimConfig(BaseModel):
    seed:            int   = 42
    dt:              float = 0.025
    t_total_ms:      float = 5000.0
    stim_period_ms:  float = 500.0
    omission_ms:     float = 2500.0
    stim_amp:        float = 2.0
    td_amp:           float = 0.5
    bu_on:           bool  = False   # False + td_on=True = omission context
    td_on:           bool  = True

class AgentRequest(BaseModel):
    prompt: str
    context: Optional[str] = None

# ── Application state ──────────────────────────────────────────────────────────

class StateVault:
    def __init__(self):
        self.networks:  Dict[str, NetBuilder] = {}
        self.reports:   Dict[str, ResultsReport] = {}
        self.tuning:    Dict[str, Any] = {}    # live GSDR state
        self.last_result: Optional[Dict] = None

vault = StateVault()

# ── FastAPI app ────────────────────────────────────────────────────────────────

app = FastAPI(title="jbiophysics API — Omission Edition", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/", response_class=HTMLResponse)
async def root():
    index_path = os.path.join(os.path.dirname(__file__), "index.html")
    with open(index_path, "r") as f:
        return f.read()

# ── Original generic endpoints ─────────────────────────────────────────────────

@app.post("/build")
async def build_network(config: NetworkConfig):
    net_id  = str(uuid.uuid4())[:8]
    builder = NetBuilder(seed=config.seed)
    for pop in config.populations:
        builder.add_population(pop.name, pop.n, pop.cell_type,
                               pop.noise_amp, pop.noise_tau)
    for conn in config.connections:
        builder.connect(conn.pre, conn.post, conn.synapse, conn.p, conn.g)
    for param in config.trainable:
        builder.make_trainable(param)
    vault.networks[net_id] = builder
    return {"net_id": net_id, "status": "constructed"}

@app.post("/simulate/{net_id}")
async def simulate_generic(net_id: str, dt: float = 0.1, t_max: float = 1000.0):
    if net_id not in vault.networks:
        raise HTTPException(status_code=404, detail="Network ID not found.")
    import jaxley as jx
    builder = vault.networks[net_id]
    net = builder.build()
    net.delete_recordings()
    net.cell("all").branch(0).loc(0.0).record("v")
    traces = jx.integrate(net, delta_t=dt, t_max=t_max)
    traces_np = np.array(traces)
    report = ResultsReport(traces=traces_np, dt=dt, t_max=t_max,
                           metadata={"net_id": net_id})
    vault.reports[net_id] = report
    fig = generate_dashboard(report, title=f"jbiophysics — {net_id}")
    return {"net_id": net_id, "status": "simulated",
            "dashboard": json.loads(fig.to_json())}

# ── NEW: V1 baseline simulation ────────────────────────────────────────────────

@app.post("/simulate/v1")
async def simulate_v1(config: V1SimConfig = Body(default_factory=V1SimConfig)):
    """Run 200-neuron V1 column for t_max ms and return traces + dashboard."""
    import jaxley as jx
    from jbiophysics.systems.networks.omission_v1_column import (
        build_v1_column, generate_sensory_input,
    )

    net, pops = build_v1_column(seed=config.seed)
    net_id = f"v1_{str(uuid.uuid4())[:6]}"

    # Record somatic voltage for all cells
    net.delete_recordings()
    net.cell("all").branch(0).loc(0.0).record("v")

    # Build stimulus current array
    t_ms  = np.arange(0, config.t_max, config.dt)
    stim_times = np.array(config.stim_times_ms)
    inp   = generate_sensory_input(t_ms, stim_times, stim_amp=config.stim_amp)

    # Apply as current clamp to L4 Pyr
    for cell_idx in pops.l4_pyr:
        net.cell(cell_idx).branch(0).loc(0.0).stimulate(
            np.column_stack([t_ms, inp])
        )

    traces = jx.integrate(net, delta_t=config.dt, t_max=config.t_max)
    traces_np = np.array(traces)

    report = ResultsReport(
        traces=traces_np, dt=config.dt, t_max=config.t_max,
        metadata={"net_id": net_id, "type": "v1_baseline", "seed": config.seed,
                  "population_offsets": {
                      "L23_Pyr": (min(pops.l23_pyr), max(pops.l23_pyr)+1),
                      "L4_Pyr":  (min(pops.l4_pyr),  max(pops.l4_pyr)+1),
                      "L56_Pyr": (min(pops.l56_pyr), max(pops.l56_pyr)+1),
                  }}
    )
    vault.reports[net_id] = report

    # Raw raster b64 for quick check
    pops_dict = {
        "l23": pops.l23_pyr, "l4": pops.l4_pyr,
        "l56": pops.l56_pyr, "pv": pops.pv,
        "sst": pops.sst,     "vip": pops.vip,
    }
    raster_b64 = plot_omission_raster(
        traces_np, config.dt, pops_dict,
        title=f"V1 Baseline Raster — {net_id}",
    )

    return {
        "net_id": net_id,
        "status": "simulated",
        "n_cells": int(traces_np.shape[0]),
        "n_steps": int(traces_np.shape[1]),
        "raster_b64": raster_b64,
    }

# ── NEW: Two-column omission simulation ───────────────────────────────────────

@app.post("/simulate/omission")
async def simulate_omission(config: OmissionSimConfig = Body(default_factory=OmissionSimConfig)):
    """
    Run 5000ms two-column omission trial.
    Default: BU=OFF, TD=ON  (omission context 3).
    """
    import jaxley as jx
    from jbiophysics.systems.networks.omission_two_column import (
        build_omission_network, OmissionTrialConfig,
        make_context_inputs, extract_lfp, detect_spikes,
    )

    onet = build_omission_network(seed=config.seed)
    net  = onet.net
    net_id = f"omission_{str(uuid.uuid4())[:6]}"

    # Record all cells
    net.delete_recordings()
    net.cell("all").branch(0).loc(0.0).record("v")

    trial_cfg = OmissionTrialConfig(
        t_total_ms=config.t_total_ms,
        dt_ms=config.dt,
        stim_period_ms=config.stim_period_ms,
        omission_ms=config.omission_ms,
        stim_amp=config.stim_amp,
        td_amp=config.td_amp,
        bu_on=config.bu_on,
        td_on=config.td_on,
    )
    n_total = onet.n_v1 + onet.n_ho
    currents = make_context_inputs(trial_cfg, onet.v1_pops, onet.ho_pops, onet.n_v1)

    # Inject per-cell stimuli
    t_ms = np.arange(0, config.t_total_ms, config.dt)
    for cell_idx in range(n_total):
        if np.any(currents[cell_idx] != 0.0):
            net.cell(cell_idx).branch(0).loc(0.0).stimulate(
                np.column_stack([t_ms, currents[cell_idx]])
            )

    traces = jx.integrate(net, delta_t=config.dt, t_max=config.t_total_ms)
    traces_np = np.array(traces)

    # LFP extraction
    lfp_v1 = extract_lfp(traces_np, onet.v1_pops.l23_pyr + onet.v1_pops.l56_pyr)
    lfp_ho = extract_lfp(traces_np, onet.ho_pops.l23_pyr + onet.ho_pops.l56_pyr)

    # Spike detection
    spikes = detect_spikes(traces_np)
    spike_counts = {k: len(v) for k, v in spikes.items()}

    # Visualizations
    pops_dict = {
        "l23": onet.v1_pops.l23_pyr, "l4": onet.v1_pops.l4_pyr,
        "l56": onet.v1_pops.l56_pyr, "pv": onet.v1_pops.pv,
        "sst": onet.v1_pops.sst,     "vip": onet.v1_pops.vip,
        "ho":  onet.ho_pops.all_pyr,
    }
    raster_b64 = plot_omission_raster(
        traces_np, config.dt, pops_dict,
        omission_onset_ms=config.omission_ms if not config.bu_on else None,
        title=f"Omission Raster (BU={config.bu_on}, TD={config.td_on})",
    )
    lfp_b64 = plot_lfp_traces(
        lfp_v1, lfp_ho, config.dt,
        omission_onset_ms=config.omission_ms if not config.bu_on else None,
    )
    fs = 1000.0 / config.dt
    tfr_b64 = plot_tfr(lfp_v1, config.dt, title="V1 LFP TFR — Omission Context")

    # Store for /visualize
    vault.last_result = {
        "net_id": net_id,
        "raster_b64": raster_b64,
        "lfp_b64": lfp_b64,
        "tfr_b64": tfr_b64,
    }

    return {
        "net_id": net_id,
        "status": "simulated",
        "context": {"bu_on": config.bu_on, "td_on": config.td_on},
        "n_cells": int(traces_np.shape[0]),
        "n_steps": int(traces_np.shape[1]),
        "total_spikes": int(sum(spike_counts.values())),
        "raster_b64": raster_b64,
        "lfp_b64": lfp_b64,
        "tfr_b64": tfr_b64,
    }

# ── NEW: Tuning endpoints ──────────────────────────────────────────────────────

@app.get("/tuning/status")
async def tuning_status():
    """Live GSDR epoch and loss. Populated while /tuning/run is active."""
    status = vault.tuning.get("status", {})
    if not status:
        return {"status": "idle", "epoch": 0, "loss": None}
    return {"status": "running", **status}

@app.get("/tuning/metrics")
async def tuning_metrics():
    """Full history of RSA, Kappa, SSS per epoch."""
    history = vault.tuning.get("history", [])
    return {"n_epochs": len(history), "history": history}

@app.post("/tuning/run")
async def tuning_run(
    n_epochs: int = 30,
    context_bu: bool = False,
    context_td: bool = True,
    seed: int = 42,
):
    """
    Kick off GSDR tuning in a background thread (non-blocking).
    Poll /tuning/status for live progress.
    """
    from jbiophysics.systems.networks.omission_two_column import (
        build_omission_network, OmissionTrialConfig,
        make_context_inputs, extract_lfp, detect_spikes,
    )
    from jbiophysics.core.optimizers.omission_objective import OmissionLoss, run_gsdr_tuning
    from jbiophysics.core.optimizers.omission_metrics import empirical_omission_target_lfp

    def _run():
        import jaxley as jx
        vault.tuning["status"]  = {"epoch": 0, "loss": None}
        vault.tuning["history"] = []

        onet = build_omission_network(seed=seed)
        net  = onet.net
        net.delete_recordings()
        net.cell("all").branch(0).loc(0.0).record("v")

        trial_cfg = OmissionTrialConfig(
            t_total_ms=1000.0, dt_ms=0.025, bu_on=context_bu, td_on=context_td
        )
        n_total  = onet.n_v1 + onet.n_ho
        currents = make_context_inputs(trial_cfg, onet.v1_pops, onet.ho_pops, onet.n_v1)
        t_ms     = np.arange(0, 1000.0, 0.025)
        for cell_idx in range(n_total):
            if np.any(currents[cell_idx] != 0.0):
                net.cell(cell_idx).branch(0).loc(0.0).stimulate(
                    np.column_stack([t_ms, currents[cell_idx]])
                )

        fs      = 1000.0 / 0.025
        n_steps = int(1000.0 / 0.025)
        lfp_tgt = empirical_omission_target_lfp(t_ms, fs)
        spikes_tgt: Dict = {}  # no target spikes (unsupervised LFP mode)

        layers = {
            "L23": onet.v1_pops.l23_pyr,
            "L4":  onet.v1_pops.l4_pyr,
            "L56": onet.v1_pops.l56_pyr,
        }
        pop_indices = onet.v1_pops.all_pyr

        def simulate_fn(net_, params):
            tr = np.array(jx.integrate(net_, params=params, delta_t=0.025, t_max=1000.0))
            sp = detect_spikes(tr)
            lf = extract_lfp(tr, onet.v1_pops.l23_pyr + onet.v1_pops.l56_pyr)
            return tr, sp, lf

        loss_fn = OmissionLoss(alpha=1.0, beta=0.5, gamma=2.0)
        state   = run_gsdr_tuning(
            net=net,
            simulate_fn=simulate_fn,
            loss_fn=loss_fn,
            layers=layers,
            pop_indices=pop_indices,
            spikes_tgt=spikes_tgt,
            lfp_tgt=lfp_tgt,
            n_steps=n_steps,
            fs=fs,
            n_epochs=n_epochs,
            seed=seed,
            status_store=vault.tuning["status"],
        )
        vault.tuning["history"] = state.history
        vault.tuning["status"]["done"] = True

    t = threading.Thread(target=_run, daemon=True)
    t.start()
    return {"status": "started", "n_epochs": n_epochs}

# ── NEW: Visualization endpoint ────────────────────────────────────────────────

@app.get("/visualize")
async def visualize():
    """Return Base64 PNGs from last simulation. Run /simulate/omission first."""
    if vault.last_result is None:
        raise HTTPException(status_code=404,
                            detail="No simulation result cached. Run /simulate/omission first.")
    return vault.last_result

# ── Original agent endpoint ────────────────────────────────────────────────────

@app.post("/agent/ask")
async def agent_ask(req: AgentRequest):
    sys.path.insert(0, "/Users/hamednejat/workspace/HNXJ/hnxj-gemini")
    try:
        from qwen_subagent import call_qwen
        response = call_qwen(f"CONTEXT: {req.context or ''}\n\nUSER: {req.prompt}")
    except ImportError:
        response = "[Agent not available in this environment]"
    return {"agent_response": response}

# ── Entrypoint ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7701)
