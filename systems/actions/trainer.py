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

from core.optimizers.AGSDR import AGSDR
from .robust_pipeline import calculate_spike_train, calculate_firing_rate, calculate_kappa, apply_global_stability

def get_integrated_loss_fn(network: jx.Network, dt: float, target_fr: float = 15.0):
    """Loss function with metabolic cost and stability hardwires."""
    def loss_fn(params: Dict[str, Any]):
        network.cell('all').branch(0).loc(0.0).record('v')
        # Step 5: Simulation with internal clipping
        voltages = jx.integrate(network, t_max=1000, delta_t=dt, params=params)
        voltages = jnp.clip(jnp.nan_to_num(voltages, nan=0.0), -100.0, 100.0)
        
        spike_train = calculate_spike_train(voltages)
        firing_rate = calculate_firing_rate(spike_train, dt)
        kappa = calculate_kappa(spike_train)
        
        loss_fr = jnp.square(firing_rate - target_fr)
        loss_kappa = kappa
        
        # Metabolic Cost
        g_total = 0.0
        for group in params:
            for k, v in group.items():
                if k.startswith('g'):
                    g_total += jnp.sum(jnp.square(jnp.nan_to_num(v)))
        
        total_loss = (loss_fr * 0.1) + (loss_kappa * 100.0) + (g_total * 0.001)
        
        aux = {"loss": total_loss, "kappa": kappa, "firing_rate": firing_rate}
        return total_loss, aux
    return loss_fn

def run_trainer_checkup(jitted_grad_fn, opt_params, trials=10):
    """10-trial pretraining checkup."""
    print(f"🔍 Trainer Pretraining Checkup...")
    for i in range(trials):
        (loss_val, aux), _ = jitted_grad_fn(opt_params)
        if jnp.isnan(loss_val) or aux['firing_rate'] > 100.0:
            print(f"   Checkup Failed (Trial {i}): FR={aux['firing_rate']:.2f}")
            return False
    print("   Checkup Passed.")
    return True

def run_pausable_training(
    net: jx.Network,
    opt_params: Any,
    epochs: int = 200,
    lr: float = 1e-3,
    save_path: str = "training_checkpoint.pkl",
    resume: bool = False
):
    """Pausable training loop with per-epoch saving."""
    dt = 0.1
    loss_fn = get_integrated_loss_fn(net, dt)
    
    # Optimizer initialization
    inner_opt = optax.adam(learning_rate=lr)
    optimizer = optax.chain(optax.clip_by_global_norm(1.0), inner_opt)
    
    if resume and os.path.exists(save_path):
        print(f"📂 Resuming from {save_path}...")
        with open(save_path, "rb") as f:
            checkpoint = pickle.load(f)
            opt_params = checkpoint['params']
            opt_state = checkpoint['state']
            start_epoch = checkpoint['epoch']
    else:
        opt_state = optimizer.init(opt_params)
        start_epoch = 0

    jitted_grad_fn = jax.jit(jax.value_and_grad(loss_fn, has_aux=True))
    
    if not resume:
        if not run_trainer_checkup(jitted_grad_fn, opt_params):
            return opt_params

    print("\n🚀 Trainer Loop Started...")
    for epoch in range(start_epoch, epochs):
        (loss_val, aux), grads = jitted_grad_fn(opt_params)
        
        updates, opt_state = optimizer.update(grads, opt_state, opt_params)
        opt_params = optax.apply_updates(opt_params, updates)
        
        # Step 5: Parameter Stability
        opt_params = apply_global_stability(opt_params)
        
        if epoch % 5 == 0:
            print(f"Epoch {epoch:03d} | Loss: {loss_val:.4f} | Kappa: {aux['kappa']:.4f} | FR: {aux['firing_rate']:.2f} Hz")
            # Save checkpoint
            with open(save_path, "wb") as f:
                pickle.dump({'params': opt_params, 'state': opt_state, 'epoch': epoch + 1}, f)
                
    print("✅ Trainer Concluded.")
    return opt_params
