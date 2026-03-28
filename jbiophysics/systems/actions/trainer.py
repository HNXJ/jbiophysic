import os
import sys
import jax
import jax.numpy as jnp
import optax
import jaxley as jx
from typing import Any, Dict, List, Optional, Tuple, Callable
import numpy as np
import pickle

# --- Path Setup ---
sys.path.insert(0, '/Users/hamednejat/workspace/Repositories/AAE')
sys.path.insert(0, '/Users/hamednejat/workspace/Repositories/jbiophys')

from jbiophysics.core.optimizers.AGSDR import AGSDR
from .robust_pipeline import calculate_spike_train, calculate_firing_rate, calculate_kappa, apply_global_stability

def calculate_biophysical_metrics(network, voltages, dt, meta):
    """Calculates extended biophysical logs (Energy, Per-Area FR)."""
    # 1. Electromagnetic Energy (Relative Energy = integral of |V * I_axial|)
    # This is a proxy for metabolic cost based on membrane and axial currents.
    # For simplicity, we approximate integrated power using voltage variance as a proxy 
    # if currents aren't explicitly recorded, but here we can integrate V^2.
    energy = jnp.sum(jnp.square(voltages)) * dt * 1e-6 

    # 2. Per-Area Firing Rate
    # Requires meta-data to identify cell indices per area
    per_area_fr = {}
    if meta:
        areas = list(set([m.get('area', 'default') for m in meta]))
        for area in areas:
            indices = [i for i, m in enumerate(meta) if m.get('area') == area]
            if indices:
                spikes = calculate_spike_train(voltages[indices, :])
                per_area_fr[area] = calculate_firing_rate(spikes, dt)
    
    return energy, per_area_fr

def get_standardized_loss_fn(network: jx.Network, dt: float, target_fr: float = 15.0, t_max: float = 1000.0, meta: List = None):
    """Refined loss with expanded aux for logging and memory efficiency."""
    
    num_neurons = len(list(network.cells))
    # Memory Optimization: For large networks and long simulations, record from a representative subset
    # 100 neurons is usually enough for a stable firing rate and kappa estimate.
    if num_neurons > 100:
        np.random.seed(42) # Deterministic subset
        subset_indices = np.random.choice(num_neurons, size=100, replace=False).tolist()
        recording_view = network.cell(subset_indices).branch(0).loc(0.0)
        print(f"🧠 Memory Optimization: Recording from a subset of {len(subset_indices)} neurons for the loss.")
    else:
        recording_view = network.cell('all').branch(0).loc(0.0)
        subset_indices = None

    def loss_fn(params: Dict[str, Any]):
        recording_view.record('v')
        # Integrate and get recorded traces
        voltages = jx.integrate(network, t_max=t_max, delta_t=dt, params=params)
        voltages = jnp.clip(jnp.nan_to_num(voltages, nan=0.0), -100.0, 100.0)
        
        # Use full voltages for energy (it's already reduced by sum)
        # Use the recorded subset for firing rate and kappa if applicable
        spike_train = calculate_spike_train(voltages)
        firing_rate = calculate_firing_rate(spike_train, dt)
        kappa = calculate_kappa(spike_train)
        energy, per_area_fr = calculate_biophysical_metrics(network, voltages, dt, meta)
        
        loss_fr = jnp.square(firing_rate - target_fr)
        loss_kappa = kappa
        
        # Metabolic cost from weights
        g_total = 0.0
        for group in params:
            for k, v in group.items():
                if k.startswith('g'):
                    g_total += jnp.sum(jnp.square(jnp.nan_to_num(v)))
        
        total_loss = (loss_fr * 0.1) + (loss_kappa * 100.0) + (g_total * 0.001)
        
        aux = {
            "loss": total_loss, 
            "kappa": kappa, 
            "firing_rate": firing_rate,
            "energy": energy,
            "per_area_fr": per_area_fr
        }
        return total_loss, aux
    return loss_fn

def run_standardized_trainer(
    net: jx.Network,
    meta: List[Dict],
    epochs: int = 200,
    lr: float = 1e-3,
    snapshot_freq: int = 10,
    t_max: float = 1000.0,
    method: str = "Adam",
    target_fr: float = 15.0
):
    """
    Standardized Trainer Pipeline.
    Returns: (ParamsList, Labels, TrainingLog)
    """
    dt = 0.1
    loss_fn = get_standardized_loss_fn(net, dt, t_max=t_max, meta=meta, target_fr=target_fr)
    
    # Initialize Optimizer
    if method == "AGSDR":
        inner_opt = optax.adam(learning_rate=lr)
        optimizer = AGSDR(inner_optimizer=inner_opt) # Assuming robust defaults
    else:
        optimizer = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(learning_rate=lr))

    opt_params = net.get_parameters()
    opt_state = optimizer.init(opt_params)
    jitted_grad_fn = jax.jit(jax.value_and_grad(loss_fn, has_aux=True))
    
    params_snapshots = [opt_params]
    labels = ["t-0"]
    training_log = {
        "loss": [], "kappa": [], "firing_rate": [], "energy": [], 
        "per_area_fr": [], "alpha": [], "deselections": [], "param_change": []
    }

    print(f"🚀 Standardized Trainer: Optimizing {epochs} trials...")
    
    prev_params = opt_params
    deselections = 0
    import gc

    for epoch in range(1, epochs + 1):
        (loss_val, aux), grads = jitted_grad_fn(opt_params)
        
        # Track Alpha and Deselections if using AGSDR
        current_alpha = getattr(opt_state, 'a_opt', 0.5) if method == "AGSDR" else 0.0
        # Placeholder for deselection logic tracking
        
        updates, opt_state = optimizer.update(grads, opt_state, opt_params)
        opt_params = optax.apply_updates(opt_params, updates)
        opt_params = apply_global_stability(opt_params)
        
        # Calculate Relative Param Change
        flat_new, _ = jax.flatten_util.ravel_pytree(opt_params)
        flat_old, _ = jax.flatten_util.ravel_pytree(prev_params)
        change = jnp.linalg.norm(flat_new - flat_old) / (jnp.linalg.norm(flat_old) + 1e-8)
        prev_params = opt_params

        # Update Log
        training_log["loss"].append(float(loss_val))
        training_log["kappa"].append(float(aux["kappa"]))
        training_log["firing_rate"].append(float(aux["firing_rate"]))
        training_log["energy"].append(float(aux["energy"]))
        training_log["per_area_fr"].append(aux["per_area_fr"])
        training_log["alpha"].append(float(current_alpha))
        training_log["param_change"].append(float(change))
        
        if epoch % snapshot_freq == 0:
            params_snapshots.append(opt_params)
            labels.append(f"t-{epoch}")
            print(f"Trial {epoch:03d} | Loss: {loss_val:.4f} | FR: {aux['firing_rate']:.2f}Hz | Change: {change:.2e}")
        
        # Memory Management
        if epoch % 5 == 0:
            gc.collect()

    print("✅ Training Concluded.")
    return params_snapshots, labels, training_log
