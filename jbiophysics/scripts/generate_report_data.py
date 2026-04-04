import argparse
import os
import sys
import numpy as np
import jaxley as jx
import matplotlib.pyplot as plt
import base64
import json
from dataclasses import asdict

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from jbiophysics.systems.networks.omission_two_column import (
    build_omission_network, OmissionTrialConfig,
    make_context_inputs, extract_lfp, detect_spikes,
)
from jbiophysics.viz.omission_viz import (
    plot_omission_raster, plot_lfp_traces, plot_tfr
)

def run_context_sim(onet, context_name, bu_on, td_on, t_total=2000.0, dt=0.025):
    print(f"--- Context: {context_name} (BU={bu_on}, TD={td_on}) ---")
    net = onet.net
    
    config = OmissionTrialConfig(
        t_total_ms=t_total,
        dt_ms=dt,
        bu_on=bu_on,
        td_on=td_on,
        td_amp=0.5 if td_on else 0.0
    )
    
    net.delete_recordings()
    net.cell("all").branch(0).loc(0.0).record("v")
    
    currents = make_context_inputs(config, onet.v1_pops, onet.ho_pops, onet.n_v1)
    for cell_idx in range(currents.shape[0]):
        if np.any(currents[cell_idx] != 0.0):
            net.cell(cell_idx).branch(0).loc(0.0).stimulate(currents[cell_idx])
            
    traces = jx.integrate(net, delta_t=dt, t_max=t_total)
    traces_np = np.array(traces)
    
    # Analysis
    lfp_v1 = extract_lfp(traces_np, onet.v1_pops.l23_pyr + onet.v1_pops.l56_pyr)
    lfp_ho = extract_lfp(traces_np, onet.ho_pops.l23_pyr + onet.ho_pops.l56_pyr)
    spikes = detect_spikes(traces_np)
    
    v1_spikes = sum(len(v) for k, v in spikes.items() if k < onet.n_v1)
    ho_spikes  = sum(len(v) for k, v in spikes.items() if k >= onet.n_v1)
    
    mfr_v1 = v1_spikes / onet.n_v1 / (t_total/1000.0)
    mfr_ho = ho_spikes / onet.n_ho / (t_total/1000.0)
    
    # Visuals
    pops_dict = {
        "l23": onet.v1_pops.l23_pyr, "l4": onet.v1_pops.l4_pyr,
        "l56": onet.v1_pops.l56_pyr, "pv": onet.v1_pops.pv,
        "sst": onet.v1_pops.sst,     "vip": onet.v1_pops.vip,
        "ho":  onet.ho_pops.all,
    }
    
    raster_b64 = plot_omission_raster(traces_np, dt, pops_dict, title=f"Raster: {context_name}")
    lfp_b64    = plot_lfp_traces(lfp_v1, lfp_ho, dt, title=f"LFP: {context_name}")
    tfr_b64    = plot_tfr(lfp_v1, dt, title=f"TFR (V1): {context_name}")
    
    return {
        "context": context_name,
        "mfr_v1": mfr_v1,
        "mfr_ho": mfr_ho,
        "raster": raster_b64,
        "lfp": lfp_b64,
        "tfr": tfr_b64
    }

def main():
    parser = argparse.ArgumentParser(description="Generate context simulation data.")
    parser.add_argument("--output_dir", type=str, default="results/data",
                        help="Directory to save the output files.")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    onet = build_omission_network(seed=42)
    
    contexts = [
        ("FF_Only",     True,  False),
        ("Spontaneous", False, False),
        ("Attended",    True,  True),
        ("Omission",    False, True),
    ]
    
    results = []
    for name, bu, td in contexts:
        res = run_context_sim(onet, name, bu, td)
        results.append(res)
        
        # Save individual images for Markdown report
        for img_type in ["raster", "lfp", "tfr"]:
            path = os.path.join(args.output_dir, f"{name.lower()}_{img_type}.png")
            with open(path, "wb") as f:
                f.write(base64.b64decode(res[img_type]))
                
    # Save metrics JSON
    metrics_to_save = [{k: v for k, v in r.items() if k not in ["raster", "lfp", "tfr"]} for r in results]
    metrics_path = os.path.join(args.output_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics_to_save, f, indent=2)
        
    print(f"✅ Full Context Simulation Data Generated in '{args.output_dir}'")

if __name__ == "__main__":
    main()
