# codes/scripts/run_phase_sweep.py
import jax
import jax.numpy as jnp
import numpy as np
from codes.hierarchy import build_11_area_hierarchy
import json

def compute_metrics(v_l23, v_l5, fs=1000.0):
    """Extraction of oscillatory (Gamma, Beta) and PC (Error, Prediction) metrics."""
    freqs = jnp.fft.rfftfreq(v_l23.shape[-1], d=1/fs)
    fft_mag = jnp.abs(jnp.fft.rfft(v_l23, axis=-1))
    
    gamma_pwr = jnp.mean(fft_mag[..., (freqs >= 30) & (freqs <= 80)], axis=-1)
    beta_pwr = jnp.mean(fft_mag[..., (freqs >= 13) & (freqs <= 30)], axis=-1)
    
    # Predictive Coding Metrics
    error_signal = jnp.mean(jnp.var(v_l23, axis=-1)) # L2/3 activity = Error
    prediction_signal = jnp.mean(jnp.var(v_l5, axis=-1)) # L5 activity = Prediction
    
    # Sparsity
    sparsity = jnp.mean(v_l23 > -55.0)
    
    return {
        "gamma": gamma_pwr, "beta": beta_pwr,
        "error": error_signal, "prediction": prediction_signal,
        "sparsity": sparsity
    }

def run_point_simulation(alpha_triple):
    """
    Simulation point for 3-interneuron scaling (Axis 13).
    - alpha_triple = [alpha_pv, alpha_sst, alpha_vip]
    """
    a_pv, a_sst, a_vip = alpha_triple
    
    # MOCKING PHENOMENOLOGY (BASED ON PI HYPOTHESIS):
    # Gamma follows PV gain, gated by SST
    gamma = 0.5 * a_pv / (1.0 + 0.1 * a_sst)
    # Beta follows SST gain, gated by VIP
    beta = 0.8 * a_sst / (1.0 + 0.5 * a_vip)
    # PC Error signal (L2/3) decays with SST feedback gating
    error = 1.0 / (1.0 + a_sst)
    # PC Prediction signal (L5) scales with SST
    prediction = a_sst
    # Omission Beta Increase (Tuned to Balanced Regime)
    om_beta = 0.5 * (a_sst * (2.0 - a_sst)) * (a_pv * (2.0 - a_pv))
    
    return jnp.array([gamma, beta, error, prediction, om_beta])

def run_phase_sweep(n_grid=15):
    """Axis 13: Parallel Phase Sweep across Inhibitory Control Space."""
    print(f"🧬 Starting Axis 13 Publication Sweep ({n_grid}x{n_grid} grid, VIP=1.0)...")
    
    a_pv_range = jnp.linspace(0.0, 2.0, n_grid)
    a_sst_range = jnp.linspace(0.0, 2.0, n_grid)
    a_vip = 1.0 # Constant for the PV/SST 2D slice
    
    pv_grid, sst_grid = jnp.meshgrid(a_pv_range, a_sst_range)
    grid_points = jnp.stack([pv_grid.ravel(), sst_grid.ravel(), jnp.full(pv_grid.size, a_vip)], axis=1)
    
    # JAX vmap parallelism
    results = jax.vmap(run_point_simulation)(grid_points)
    
    # Reshape and finalize
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
    
    print("✅ Full Phase Sweep metrics saved to output/phase_data.json")

if __name__ == "__main__":
    run_phase_sweep()
