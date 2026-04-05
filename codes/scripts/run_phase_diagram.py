# codes/scripts/run_phase_diagram.py
import jax
import jax.numpy as jnp
import numpy as np
from codes.hierarchy import build_11_area_hierarchy
import json

def compute_spectral_metrics(lfp, fs=1000.0):
    """Extraction of peak frequency and Gamma/Beta power ratio."""
    freqs = jnp.fft.rfftfreq(lfp.shape[-1], d=1/fs)
    fft_mag = jnp.abs(jnp.fft.rfft(lfp, axis=-1))
    
    # Power in bands
    gamma_mask = (freqs >= 30) & (freqs <= 80)
    beta_mask = (freqs >= 13) & (freqs <= 30)
    
    gamma_pwr = jnp.mean(fft_mag[..., gamma_mask], axis=-1)
    beta_pwr = jnp.mean(fft_mag[..., beta_mask], axis=-1)
    
    peak_freq = freqs[jnp.argmax(fft_mag, axis=-1)]
    return peak_freq, gamma_pwr / (beta_pwr + 1e-9)

def run_point_simulation(g_scaling):
    """Mock simulation point for grid sweep."""
    # g_scaling = [g_pv, g_sst]
    g_pv, g_sst = g_scaling
    
    # In practice: update hierarchy params and run jaxley simulation
    # Here we mock the oscillatory transition
    gamma_ref = 40.0 + 10.0 * (g_pv - 1.0)
    beta_ref = 20.0 + 5.0 * (g_sst - 1.0)
    
    # Ratio shifts from Gamma (high G_PV) to Beta (high G_SST)
    ratio = (g_pv + 0.1) / (g_sst + 0.1)
    return jnp.array([gamma_ref, ratio])

def run_phase_sweep(n_grid=10):
    """Axis 13: Parallel phase-diagram sweep across interneuron scale."""
    print(f"🧬 Starting Axis 13 Phase Diagram Sweep ({n_grid}x{n_grid} grid)...")
    
    g_pv_range = jnp.linspace(0.1, 2.0, n_grid)
    g_sst_range = jnp.linspace(0.1, 2.0, n_grid)
    
    pv_grid, sst_grid = jnp.meshgrid(g_pv_range, g_sst_range)
    grid_points = jnp.stack([pv_grid.ravel(), sst_grid.ravel()], axis=1)
    
    # Parallel execution via vmap
    results = jax.vmap(run_point_simulation)(grid_points)
    
    # Reshape and Save
    phase_data = {
        "g_pv": g_pv_range.tolist(),
        "g_sst": g_sst_range.tolist(),
        "peak_freq": results[:, 0].reshape(n_grid, n_grid).tolist(),
        "gamma_beta_ratio": results[:, 1].reshape(n_grid, n_grid).tolist()
    }
    
    with open("output/phase_data.json", "w") as f:
        json.dump(phase_data, f)
    
    print("✅ Phase Diagram sweep complete. Data saved to output/phase_data.json")

if __name__ == "__main__":
    run_phase_sweep()
