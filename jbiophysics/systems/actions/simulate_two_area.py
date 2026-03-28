import os
import sys
# Ensure the local jbiophysics package is in path
sys.path.insert(0, '/Users/hamednejat/workspace/Computational/jbiophysics')

import jax
import jax.numpy as jnp
import jaxley as jx
import numpy as np
import matplotlib.pyplot as plt

from jbiophysics.systems.networks.laminar_column import build_laminar_column
from jbiophysics.systems.networks.inter_area import connect_cortical_areas
from jbiophysics.systems.visualizers.compute_kappa import compute_kappa
from jbiophysics.systems.visualizers.calculate_firing_rates import calculate_firing_rates

from jax import config
config.update("jax_platform_name", "cpu")

def run_two_area_sim():
    print("🚀 Building Two-Area Network (Low Cortex -> High Cortex)...")
    
    # 1. Configuration for both areas
    low_config = {'num_superficial': 20, 'num_mid': 10, 'num_deep': 20, 'seed': 42}
    high_config = {'num_superficial': 20, 'num_mid': 10, 'num_deep': 20, 'seed': 43}
    
    # 2. Build and Connect
    net, (meta_low, meta_high) = connect_cortical_areas(low_config, high_config)
    
    # 3. Simulate Baseline Activity
    dt, t_max = 0.1, 1000.0
    net.cell('all').branch(0).loc(0.0).record()
    
    print("Integrating Two-Area Model...")
    traces = jx.integrate(net, delta_t=dt, t_max=t_max)
    
    # 4. Analysis
    afr = calculate_firing_rates(traces, dt)
    
    threshold = -20.0
    spikes = (traces[:, :-1] < threshold) & (traces[:, 1:] >= threshold)
    spike_matrix = jnp.zeros_like(traces).at[:, 1:].set(spikes.astype(jnp.float32))
    kappa = compute_kappa(spike_matrix, 1000.0/dt)
    
    print(f"✅ Simulation Complete.")
    print(f"Mean AFR: {jnp.mean(afr):.2f} Hz")
    print(f"Population Kappa: {kappa:.4f}")

    # 5. Visualization (Raster)
    plt.figure(figsize=(15, 8), facecolor='white')
    time_axis = np.linspace(0, t_max, spike_matrix.shape[1])
    for i in range(spike_matrix.shape[0]):
        spike_times = time_axis[spike_matrix[i, :] > 0]
        plt.vlines(spike_times, i + 0.5, i + 1.5, color='black', linewidth=0.8)
    
    plt.axhline(y=50, color='red', linestyle='--', alpha=0.5, label='Area Boundary')
    plt.title(f"Two-Area Raster Plot | Kappa: {kappa:.4f} | Avg AFR: {jnp.mean(afr):.2f} Hz", fontweight='bold')
    plt.xlabel("Time (ms)")
    plt.ylabel("Neuron Index (0-49: Low, 50-99: High)")
    plt.legend()
    
    save_path = "/Users/hamednejat/workspace/media/figures/optimizer_tests/two_area_raster.png"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300)
    print(f"🖼️ Raster saved to: {save_path}")

if __name__ == "__main__":
    run_two_area_sim()
