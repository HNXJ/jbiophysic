import os
import sys
# Ensure the local jbiophysics package is in path
sys.path.insert(0, '/Users/hamednejat/workspace/Computational/jbiophysics')

import jax
import jax.numpy as jnp
import jaxley as jx
import optax
import numpy as np
import matplotlib.pyplot as plt

from jbiophysics.core.mechanisms.models import Inoise, GradedAMPA, GradedGABAa
from jbiophysics.core.optimizers.AGSDR import AGSDR, ClampTransform
from jbiophysics.systems.networks.laminar_column import build_laminar_column
from jbiophysics.systems.visualizers.compute_kappa import compute_kappa
from jbiophysics.systems.visualizers.calculate_firing_rates import calculate_firing_rates

from jax import config
config.update("jax_platform_name", "cpu")

def optimize_laminar():
    print("🚀 Initializing 100-neuron Laminar Column Optimization...")
    
    # 1. Build Network
    num_sup, num_mid, num_deep = 40, 20, 40
    net, meta = build_laminar_column(num_superficial=num_sup, num_mid=num_mid, num_deep=num_deep, seed=42)
    
    # 2. Setup Trainability
    net.make_trainable("gAMPA")
    net.make_trainable("gGABAa")
    
    # 3. Loss Definition
    dt, t_max = 0.1, 1000.0
    target_low, target_high = 5.0, 10.0
    
    def loss_fn(params, key):
        # Baseline noise simulation
        net.delete_recordings()
        net.cell('all').branch(0).loc(0.0).record()
        
        traces = jx.integrate(net, params=params, delta_t=dt, t_max=t_max)
        
        # AFR Loss: Target range [5, 10]
        afr = calculate_firing_rates(traces, dt)
        mean_afr = jnp.mean(afr)
        
        # Penalize outside the [5, 10] window
        afr_loss = jnp.where(mean_afr < target_low, jnp.square(target_low - mean_afr), 
                             jnp.where(mean_afr > target_high, jnp.square(mean_afr - target_high), 0.0))
        
        # Kappa Loss (target 0)
        threshold = -20.0
        spikes = (traces[:, :-1] < threshold) & (traces[:, 1:] >= threshold)
        spike_matrix = jnp.zeros_like(traces).at[:, 1:].set(spikes.astype(jnp.float32))
        kappa = compute_kappa(spike_matrix, 1000.0/dt)
        kappa_loss = jnp.abs(kappa)
        
        # Dead-network barrier
        penalty = jnp.where(mean_afr < 0.5, 100.0, 0.0)
        
        return afr_loss + 20.0 * kappa_loss + penalty, (mean_afr, kappa)

    # 4. AGSDR Optimizer
    inner = optax.adam(1e-2)
    optimizer = AGSDR(inner, checkpoint_n=20, alpha_min=0.1)
    
    opt_params = net.get_parameters()
    opt_state = optimizer.init(opt_params)
    
    val_grad_fn = jax.jit(jax.value_and_grad(loss_fn, has_aux=True))
    
    # 5. Training Loop
    num_trials = 500
    print(f"Starting 500-trial optimization loop...")
    
    for i in range(num_trials):
        key = jax.random.PRNGKey(i + 730)
        (loss_val, (afr, kappa)), grads = val_grad_fn(opt_params, key)
        
        updates, opt_state = optimizer.update(grads, opt_state, params=opt_params, value=loss_val, key=key)
        opt_params = optax.apply_updates(opt_params, updates)
        
        if i % 10 == 0:
            print(f"  Trial {i} | Loss: {loss_val:.4f} | Alpha: {opt_state.a:.4f} | Kappa: {kappa:.4f} | AFR: {afr:.2f}")

    print("✅ Optimization complete. Saving parameters...")
    import pickle
    with open("laminar_100_params.pkl", "wb") as f:
        pickle.dump(opt_params, f)

if __name__ == "__main__":
    optimize_laminar()
