import os
import sys
import numpy as np
import jaxley as jx
import matplotlib.pyplot as plt
import base64

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from jbiophysics.networks.omission_two_column import (
    build_omission_network, OmissionTrialConfig,
    make_context_inputs, extract_lfp, detect_spikes,
)
from jbiophysics.viz.omission_viz import (
    plot_omission_raster, plot_lfp_traces, plot_tfr,
    BG, GOLD, WHITE, CYAN, VIOLET
)

def run_trial(t_total=1000.0, dt=0.025, seed=42):
    print(f"🚀 Running Omission Paradigm Simulation ({t_total}ms) ...")
    
    # 1. Build Network
    onet = build_omission_network(seed=seed)
    net  = onet.net
    
    # 2. Configure Omission Context (BU=OFF, TD=ON)
    config = OmissionTrialConfig(
        t_total_ms=t_total,
        dt_ms=dt,
        bu_on=False,
        td_on=True,
        td_amp=0.5
    )
    
    # 3. Setup Stimulation
    net.delete_recordings()
    net.cell("all").branch(0).loc(0.0).record("v")
    
    t_ms = np.arange(0, t_total, dt)
    currents = make_context_inputs(config, onet.v1_pops, onet.ho_pops, onet.n_v1)
    
    for cell_idx in range(currents.shape[0]):
        if np.any(currents[cell_idx] != 0.0):
            # jaxley 0.13.0: stimulate expects (num_compartments, T) or (T,) for single loc.
            # Passing (T,) directly as currents[cell_idx] is 1D.
            net.cell(cell_idx).branch(0).loc(0.0).stimulate(
                currents[cell_idx]
            )
    
    # 4. Integrate
    print(f"⌛ Integrating 300 neurons for {len(t_ms)} steps...")
    traces = jx.integrate(net, delta_t=dt, t_max=t_total)
    traces_np = np.array(traces)
    
    # 5. Analysis
    lfp_v1 = extract_lfp(traces_np, onet.v1_pops.l23_pyr + onet.v1_pops.l56_pyr)
    lfp_ho = extract_lfp(traces_np, onet.ho_pops.l23_pyr + onet.ho_pops.l56_pyr)
    spikes = detect_spikes(traces_np)
    
    total_spikes = sum(len(v) for v in spikes.values())
    v1_spikes = sum(len(v) for k, v in spikes.items() if k < onet.n_v1)
    ho_spikes  = sum(len(v) for k, v in spikes.items() if k >= onet.n_v1)
    
    print("\n--- Simulation Metrics ---")
    print(f"Total Spikes: {total_spikes}")
    print(f"V1 Column:    {v1_spikes} spikes (~{v1_spikes/onet.n_v1/(t_total/1000.0):.1f} Hz)")
    print(f"HO Column:    {ho_spikes} spikes (~{ho_spikes/onet.n_ho/(t_total/1000.0):.1f} Hz)")
    
    # 6. Visualization
    print("\n🎨 Generating visualizations...")
    os.makedirs("results", exist_ok=True)
    
    pops_dict = {
        "l23": onet.v1_pops.l23_pyr, "l4": onet.v1_pops.l4_pyr,
        "l56": onet.v1_pops.l56_pyr, "pv": onet.v1_pops.pv,
        "sst": onet.v1_pops.sst,     "vip": onet.v1_pops.vip,
        "ho":  onet.ho_pops.all,
    }
    
    # Raster
    raster_b64 = plot_omission_raster(
        traces_np, dt, pops_dict,
        title=f"Omission Trial Verification — {t_total}ms",
    )
    with open("results/raster_omission.png", "wb") as f:
        f.write(base64.b64decode(raster_b64))
        
    # LFP
    lfp_b64 = plot_lfp_traces(lfp_v1, lfp_ho, dt, title="LFP Verification — Omission Context")
    with open("results/lfp_omission.png", "wb") as f:
        f.write(base64.b64decode(lfp_b64))
        
    # TFR
    tfr_b64 = plot_tfr(lfp_v1, dt, f_min=4, f_max=100)
    with open("results/tfr_omission.png", "wb") as f:
        f.write(base64.b64decode(tfr_b64))
        
    print("✅ Results saved to 'results/' directory.")

if __name__ == "__main__":
    run_trial()
