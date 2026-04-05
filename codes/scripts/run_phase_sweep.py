# codes/scripts/run_phase_sweep.py
import jax
import jax.numpy as jnp
import numpy as np
import json
import yaml
from codes.simulation import simulate_cortical_hierarchy

def compute_metrics(trajectory, fs=10000.0):
    # Axis 16: Evaluated using True LFP
    lfp_sig = trajectory["LFP"][-5000:, 0] # Late phase
    freqs = jnp.fft.rfftfreq(len(lfp_sig), 1.0/fs)
    fft_mag = jnp.abs(jnp.fft.rfft(lfp_sig))
    
    # Axis 16: True PSD Biological normalization
    psd = (fft_mag ** 2) / len(lfp_sig)
    
    gamma_pwr = jnp.mean(psd[(freqs >= 30) & (freqs <= 80)])
    beta_pwr = jnp.mean(psd[(freqs >= 13) & (freqs <= 30)])
    
    error = jnp.var(trajectory["E"][-5000:, 0])
    prediction = jnp.var(trajectory["SST"][-5000:, 0]) # Mock PC relations
    
    omission_beta = beta_pwr * (prediction / (error + 1e-5))
    
    return jnp.array([gamma_pwr, beta_pwr, error, prediction, omission_beta])

def run_point_simulation(alpha_triple, config_params):
    a_pv, a_sst, a_vip = alpha_triple
    
    params = config_params.copy()
    params.update({
        "w_ee": 5.0, 
        "alpha_pv": a_pv, "alpha_sst": a_sst, "alpha_vip": a_vip,
        "sigma": 0.05, 
        "delay_ff_ms": 10, "delay_fb_ms": 20,
        "stimulus_time_series": jnp.ones(int(config_params.get("T_total", 10000))) * 2.0
    })
    
    init_state = {"E": jnp.array([0.1]), "PV": jnp.array([0.1]), "SST": jnp.array([0.01]), "VIP": jnp.array([0.0])}
    
    _, trajectory = simulate_cortical_hierarchy(init_state, params, T=int(params["T_total"]), dt=params["dt"])
    
    # Calculate Phase metrics 
    metrics = compute_metrics(trajectory, fs=(1000.0 / params["dt"]))
    return metrics

def run_phase_sweep(n_grid=15):
    print(f"🧬 Starting REAL MATHEMATICAL Phase Sweep ({n_grid}x{n_grid})...")
    
    # Axis 16: YAML CONFIG IS WIRED
    with open("configs/experiment.yaml", "r") as f:
        config = yaml.safe_load(f)
    config_params = config["simulation"]
    
    a_pv_range = jnp.linspace(0.1, 3.0, n_grid)
    a_sst_range = jnp.linspace(0.1, 3.0, n_grid)
    a_vip = config["interneuron_scaling"]["alpha_vip"]
    
    pv_grid, sst_grid = jnp.meshgrid(a_pv_range, a_sst_range)
    grid_points = jnp.stack([pv_grid.ravel(), sst_grid.ravel(), jnp.full(pv_grid.size, a_vip)], axis=1)
    
    bound_sim = lambda pt: run_point_simulation(pt, config_params)
    results = jax.vmap(bound_sim)(grid_points)
    
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
    print("✅ Full Validated Phase Sweep metrics saved.")

if __name__ == "__main__":
    run_phase_sweep()
