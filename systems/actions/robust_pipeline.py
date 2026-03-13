import os
import sys
import jax
import jax.numpy as jnp
import optax
import jaxley as jx
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pickle

# --- Global Path Setup ---
# Assuming robust_pipeline is inside jbiophys/systems/actions/
# We add Repositories paths to ensure all custom modules are accessible.
sys.path.insert(0, '/Users/hamednejat/workspace/Repositories/AAE')
sys.path.insert(0, '/Users/hamednejat/workspace/Repositories/jbiophys')

from core.mechanisms.models import GradedAMPA, GradedGABAa, make_synapses_independent, get_parameter_summary
from core.optimizers.AGSDR import AGSDR

# --- Stability Hardwires & Analysis Helpers ---

@jax.custom_jvp
def sg_spike(x):
    """Surrogate Gradient Spiking function (float return)."""
    return (x > 0).astype(jnp.float32)

@sg_spike.defjvp
def sg_spike_jvp(primals, tangents):
    x, = primals
    x_dot, = tangents
    primal_out = sg_spike(x)
    # Rectangular surrogate gradient
    tangent_out = jnp.maximum(0, 1 - jnp.abs(x)) * x_dot
    return primal_out, tangent_out

def calculate_spike_train(voltage_trace, threshold=0.0):
    return sg_spike(voltage_trace - threshold)

def calculate_firing_rate(spike_train, dt):
    total_spikes = jnp.sum(spike_train)
    num_neurons = spike_train.shape[0]
    duration_sec = spike_train.shape[1] * dt / 1000.0
    return total_spikes / (num_neurons * duration_sec + 1e-8)

def calculate_kappa(spike_train, bin_size=10):
    num_neurons, num_timesteps = spike_train.shape
    trimmed_len = (num_timesteps // bin_size) * bin_size
    trimmed_spike_train = spike_train[..., :trimmed_len]
    binned_spikes = trimmed_spike_train.reshape(num_neurons, -1, bin_size).sum(axis=-1)
    
    pop_rate = jnp.mean(binned_spikes, axis=0)
    var_pop_rate = jnp.var(pop_rate)
    mean_pop_rate = jnp.mean(pop_rate)
    kappa = var_pop_rate / (mean_pop_rate + 1e-8)
    return jnp.where(mean_pop_rate == 0, 0.0, kappa)

# --- Step 1: Initialize Network ---

def initialize_network(cells: List[jx.Cell], independent_synapses: bool = True):
    """Builds the network and configures parameter independence."""
    net = jx.Network(cells)
    
    if independent_synapses:
        # We assume standard parameters used in our projects
        # This will be refined as more parameter types are added
        pass 
    
    return net

# --- Step 2: Plausibility Sweep ---

def run_plausibility_sweep(net, dt=0.1, sim_time=1000.0):
    """Checks if network firing is in the 1Hz - 100Hz range."""
    print("🌊 Running Plausibility Sweep...")
    params = net.get_parameters()
    net.cell('all').branch(0).loc(0.0).record('v')
    
    voltages = jx.integrate(net, params=params, delta_t=dt, t_max=sim_time)
    spike_train = calculate_spike_train(voltages)
    fr = calculate_firing_rate(spike_train, dt)
    
    print(f"   Baseline Firing Rate: {fr:.2f} Hz")
    
    if fr > 100.0 or fr < 1.0:
        print(f"🚨 Network is unstable ({fr:.2f} Hz). Scaling parameters required.")
        return False, fr
    
    print("✅ Plausibility Check Passed.")
    return True, fr

# --- Step 3: Training Config & Loss ---

def get_robust_loss_fn(network: jx.Network, dt: float, target_fr: float = 15.0):
    """Loss function with metabolic cost and stability hardwires."""
    def loss_fn(params: Dict[str, Any]):
        # Record only somas for efficiency
        network.cell('all').branch(0).loc(0.0).record('v')
        voltages = jx.integrate(network, t_max=1000, delta_t=dt, params=params)
        
        # Step 5: Voltage hardwire
        voltages = jnp.clip(voltages, -100.0, 100.0)
        
        spike_train = calculate_spike_train(voltages)
        firing_rate = calculate_firing_rate(spike_train, dt)
        kappa = calculate_kappa(spike_train)
        
        # Objectives
        loss_fr = jnp.square(firing_rate - target_fr)
        loss_kappa = kappa
        
        # Step 3: Metabolic Cost (Self-Inhibition)
        # Assuming params are structured as a list of dicts from get_parameters()
        g_total = 0.0
        for group in params:
            for k, v in group.items():
                if k.startswith('g'): # Synaptic conductances
                    g_total += jnp.sum(jnp.square(jnp.nan_to_num(v)))
        
        total_loss = (loss_fr * 0.1) + (loss_kappa * 100.0) + (g_total * 0.001)
        
        aux = {"loss": total_loss, "kappa": kappa, "firing_rate": firing_rate}
        return total_loss, aux
    return loss_fn

# --- Step 4: Pretraining Checkup ---

def pretraining_checkup(jitted_grad_fn, opt_params, trials=10):
    """Executes 10 trials to evaluate stability."""
    print(f"🔍 Running Pretraining Checkup ({trials} trials)...")
    for i in range(trials):
        (loss_val, aux), grads = jitted_grad_fn(opt_params)
        if jnp.isnan(loss_val) or aux['firing_rate'] > 100.0:
            print(f"🚨 Checkup Failed at trial {i}: Loss={loss_val}, FR={aux['firing_rate']:.2f}")
            return False, aux
    print("✅ Pretraining Checkup Passed.")
    return True, aux

# --- Step 5: Parameter Hardwires ---

def apply_global_stability(params):
    """NaN handling and parameter clipping."""
    for i, group in enumerate(params):
        for k, v in group.items():
            if k.startswith('g'): # Synaptic conductances
                params[i][k] = jnp.clip(jnp.nan_to_num(v, nan=0.0), 0.0, 100.0)
    return params

# --- Step 6: 64-bit Fallback ---

def enable_x64():
    print("💎 Switching to 64-bit precision for maximum stability.")
    jax.config.update("jax_enable_x64", True)

# --- Step 7: Execution Wrapper ---

def execute_robust_training(
    net: jx.Network, 
    epochs: int = 200, 
    lr: float = 1e-3, 
    target_fr: float = 15.0,
    force_x64: bool = False
):
    """Main orchestration of the 7-step pipeline."""
    if force_x64:
        enable_x64()
        
    dt = 0.1
    # Ensure independent synapses
    make_synapses_independent(net, "gAMPA")
    make_synapses_independent(net, "gGABAa")
    
    get_parameter_summary(net)
    
    # Step 2
    success, _ = run_plausibility_sweep(net, dt)
    if not success:
        print("Suggest: Adjust Inoise or connection probability before restarting.")
        # return None
    
    # Step 3
    loss_fn = get_robust_loss_fn(net, dt, target_fr)
    
    # Optimizer
    inner_opt = optax.adam(learning_rate=lr)
    optimizer = optax.chain(
        optax.clip_by_global_norm(1.0),
        inner_opt
    )
    
    opt_params = net.get_parameters()
    opt_state = optimizer.init(opt_params)
    jitted_grad_fn = jax.jit(jax.value_and_grad(loss_fn, has_aux=True))
    
    # Step 4
    checkup_passed, _ = pretraining_checkup(jitted_grad_fn, opt_params)
    if not checkup_passed:
        print("Action: Attempting training with lower learning rate...")
        # Recurse or adjust logic here
    
    print("\n🚀 Starting Full Training Loop...")
    for epoch in range(epochs):
        (loss_val, aux), grads = jitted_grad_fn(opt_params)
        
        if jnp.isnan(loss_val):
            print(f"🚨 NaN detected at epoch {epoch}. Stability hardwires triggered.")
            
        updates, opt_state = optimizer.update(grads, opt_state, opt_params)
        opt_params = optax.apply_updates(opt_params, updates)
        
        # Step 5
        opt_params = apply_global_stability(opt_params)
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch:03d} | Loss: {loss_val:.4f} | Kappa: {aux['kappa']:.4f} | FR: {aux['firing_rate']:.2f} Hz")
            
    print("✅ Training Concluded.")
    return opt_params
