import os
import sys
# Ensure the local jbiophysics package is in path
sys.path.insert(0, '/Users/hamednejat/workspace/Computational/jbiophysics')

import jax
import jax.numpy as jnp
import jaxley as jx
import numpy as np
import pandas as pd

from jbiophysics.systems.networks.inter_area import connect_cortical_areas
from jbiophysics.systems.actions.sweep_tool import run_parameter_sweep
from jbiophysics.systems.visualizers.compute_kappa import compute_kappa
from jbiophysics.systems.visualizers.calculate_firing_rates import calculate_firing_rates

from jax import config
config.update("jax_platform_name", "cpu")

def inter_area_sim_logic(g_ff, g_fb):
    """
    Constructs and simulates the two-area model for specific conductances.
    Returns (Target Area AFR, Target Area Kappa).
    """
    low_config = {'num_superficial': 20, 'num_mid': 10, 'num_deep': 20, 'seed': 42}
    high_config = {'num_superficial': 20, 'num_mid': 10, 'num_deep': 20, 'seed': 43}
    
    # 1. Build and Connect
    net, (meta_low, meta_high) = connect_cortical_areas(low_config, high_config, g_ff=g_ff, g_fb=g_fb)
    
    # 2. Simulate
    dt, t_max = 0.1, 1000.0
    net.cell('all').branch(0).loc(0.0).record()
    traces = jx.integrate(net, delta_t=dt, t_max=t_max)
    
    # 3. Analyze TARGET AREA ONLY (High Cortex: neurons 50-99)
    target_traces = traces[50:100, :]
    afr = calculate_firing_rates(target_traces, dt)
    
    threshold = -20.0
    spikes = (target_traces[:, :-1] < threshold) & (target_traces[:, 1:] >= threshold)
    spike_matrix = jnp.zeros_like(target_traces).at[:, 1:].set(spikes.astype(jnp.float32))
    kappa = compute_kappa(spike_matrix, 1000.0/dt)
    
    return float(jnp.mean(afr)), float(kappa)

def execute_sweep():
    # Define Manifold: Mean +- std approx
    # Baseline: g_ff=0.5, g_fb=0.3
    g_ff_range = [0.1, 0.5, 1.0, 2.0]
    g_fb_range = [0.1, 0.3, 0.6, 1.2]
    
    results = []
    
    print(f"🚀 Starting 16-point Inter-Area Manifold Sweep...")
    for ff in g_ff_range:
        for fb in g_fb_range:
            print(f"  Testing g_ff={ff}, g_fb={fb}...")
            afr, kappa = inter_area_sim_logic(ff, fb)
            results.append({'g_ff': ff, 'g_fb': fb, 'target_afr': afr, 'target_kappa': kappa})
            print(f"    -> AFR: {afr:.2f} Hz | Kappa: {kappa:.4f}")

    # Save
    df = pd.DataFrame(results)
    save_path = "/Users/hamednejat/workspace/Computational/jbiophysics/systems/actions/inter_area_sweep_results.csv"
    df.to_csv(save_path, index=False)
    print(f"✨ Sweep complete. Results saved to {save_path}")

if __name__ == "__main__":
    execute_sweep()
