import os
import sys
# Ensure the local jbiophysics package is in path
sys.path.insert(0, '/Users/hamednejat/workspace/Computational/jbiophysics')

import jax
import jax.numpy as jnp
import jaxley as jx
import numpy as np
import pandas as pd
import time

from systems.networks.laminar_column import build_laminar_cells
from systems.networks.inter_area import connect_cortical_areas
from core.mechanisms.IPnoise import IPnoise
from systems.visualizers.compute_kappa import compute_kappa
from systems.visualizers.calculate_firing_rates import calculate_firing_rates

from jax import config
config.update("jax_platform_name", "cpu")

def run_ipnoise_sweep():
    print("🚀 Initializing 30-point IPnoise Manifold Sweep...")
    
    save_path = "/Users/hamednejat/workspace/Computational/jbiophysics/systems/actions/ipnoise_sweep_results.csv"
    amp_range = [0.01, 0.1, 0.5, 1.0, 3.3, 10.0]
    lambda_range = [20, 50, 100, 200, 500]
    
    results = []
    dt, t_max = 0.1, 1000.0
    
    # Pre-build base network to save time
    low_config = {'num_superficial': 20, 'num_mid': 10, 'num_deep': 20, 'seed': 42}
    high_config = {'num_superficial': 20, 'num_mid': 10, 'num_deep': 20, 'seed': 43}
    
    for amp in amp_range:
        for l in lambda_range:
            print(f"  Testing: amp={amp} nA, lambda={l} ms...")
            
            # 1. Build and Connect Areas
            net, (m_low, m_high) = connect_cortical_areas(low_config, high_config)
            
            # 2. Inject IPnoise into all cells of the final network
            net.cell('all').insert(IPnoise())
            
            # 3. Parameter setup
            # Zero out standard Inoise to see pure IPnoise effect
            try:
                net.Inoise.set("amp_noise", 0.0)
            except AttributeError:
                pass
            
            net.IPnoise.set("pulse_amp", amp)
            net.IPnoise.set("poisson_l", l)
            current_opt_params = net.get_parameters()
            
            # 4. Simulate
            net.cell('all').branch(0).loc(0.0).record()
            start_t = time.time()
            traces = jx.integrate(net, params=current_opt_params, delta_t=dt, t_max=t_max)
            duration = time.time() - start_t

            # 4. Analysis
            afr = calculate_firing_rates(traces, dt)
            threshold = -20.0
            spikes = (traces[:, :-1] < threshold) & (traces[:, 1:] >= threshold)
            spike_matrix = jnp.zeros_like(traces).at[:, 1:].set(spikes.astype(jnp.float32))
            kappa = compute_kappa(spike_matrix, 1000.0/dt)
            
            results.append({
                'pulse_amp': amp,
                'poisson_l': l,
                'mean_afr': float(jnp.mean(afr)),
                'kappa': float(kappa),
                'sim_duration': duration
            })
            print(f"    -> AFR: {jnp.mean(afr):.2f} | Kappa: {kappa:.4f}")
            
            # Incremental Save
            pd.DataFrame(results).to_csv(save_path, index=False)

    # 5. Save
    df = pd.DataFrame(results)
    save_path = "/Users/hamednejat/workspace/Computational/jbiophysics/systems/actions/ipnoise_sweep_results.csv"
    df.to_csv(save_path, index=False)
    print(f"✨ Sweep complete. Results saved to {save_path}")

if __name__ == "__main__":
    run_ipnoise_sweep()
