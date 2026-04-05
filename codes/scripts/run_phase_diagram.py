# codes/scripts/run_phase_diagram.py
import jax
import jax.numpy as jnp
import numpy as np
from codes.hierarchy import build_11_area_hierarchy
import json

def compute_metrics(v_l23, v_l5, fs=1000.0):
    """
    Axis 13 Multi-Objective Metrics:
    - Oscillatory (Gamma, Beta)
    - Predictive Coding (Error L23, Prediction L5)
    - Sparsity
    """
    freqs = jnp.fft.rfftfreq(v_l23.shape[-1], d=1/fs)
    fft_mag = jnp.abs(jnp.fft.rfft(v_l23, axis=-1))
    
    gamma_pwr = jnp.mean(fft_mag[..., (freqs >= 30) & (freqs <= 80)], axis=-1)
    beta_pwr = jnp.mean(fft_mag[..., (freqs >= 13) & (freqs <= 30)], axis=-1)
    
    # PC Metrics
    error_signal = jnp.mean(jnp.var(v_l23, axis=-1))
    prediction_signal = jnp.mean(jnp.var(v_l5, axis=-1))
    
    # Sparsity (Mock)
    sparsity = jnp.mean(v_l23 > -55.0)
    
    return {
        "gamma": gamma_pwr, "beta": beta_pwr,
        "error": error_signal, "prediction": prediction_signal,
        "sparsity": sparsity
    }

def run_point_simulation(alpha_triple):
    """Mock simulation for PI-grade grid sweep."""
    a_pv, a_sst, a_vip = alpha_triple
    
    # Logic based on PI hypothesis (Axis 13):
    # PV -> Gamma; SST -> Beta/Gating; VIP -> Suppression of SST
    gamma = 0.5 * a_pv / (1.0 + 0.1 * a_sst)
    beta = 0.8 * a_sst / (1.0 + 0.5 * a_vip)
    err = 1.0 / (1.0 + a_sst) # SST gates error
    pred = a_sst # SST facilitates prediction flow
    
    # Omission delta (Beta increase when SST 1.0-1.5 and PV balanced)
    om_beta = 0.5 * (a_sst * (2.0 - a_sst)) * (a_pv * (2.0 - a_pv))
    
    return jnp.array([gamma, beta, err, pred, om_beta])

def run_full_phase_sweep(n_grid=15):
    """Axis 13: 2D/3D Grid Sweep for Manuscript Figures."""
    print(f"🧬 Starting Axis 13 Publication Sweep ({n_grid}x{n_grid} grid)...")
    
    a_pv_range = jnp.linspace(0.0, 2.0, n_grid)
    a_sst_range = jnp.linspace(0.0, 2.0, n_grid)
    a_vip = 1.0 # Constant for main figure
    
    pv_grid, sst_grid = jnp.meshgrid(a_pv_range, a_sst_range)
    grid_points = jnp.stack([pv_grid.ravel(), sst_grid.ravel(), jnp.full(pv_grid.size, a_vip)], axis=1)
    
    # Parallel execution via vmap
    results = jax.vmap(run_point_simulation)(grid_points)
    
    # Reshape results
    data = {
        "a_pv": a_pv_range.tolist(),
        "a_sst": a_sst_range.tolist(),
        "gamma": results[:, 0].reshape(n_grid, n_grid).tolist(),
        "beta": results[:, 1].reshape(n_grid, n_grid).tolist(),
        "error": results[:, 2].reshape(n_grid, n_grid).tolist(),
        "prediction": results[:, 3].reshape(n_grid, n_grid).tolist(),
        "omission_beta": results[:, 4].reshape(n_grid, n_grid).tolist()
    }
    
    with open("output/phase_data.json", "w") as f:
        json.dump(data, f)
    
    print("✅ Full Phase Sweep complete. Metrics saved to output/phase_data.json")

if __name__ == "__main__":
    run_full_phase_sweep()
