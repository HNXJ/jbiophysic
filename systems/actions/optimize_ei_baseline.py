import os
import sys
# Ensure the local jbiophys package is in path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

import jax
import jax.numpy as jnp
import jaxley as jx
import optax
import numpy as np
import matplotlib.pyplot as plt

# Import from the local jbiophys package
from core.mechanisms.models import Inoise, GradedAMPA, GradedGABAa
from core.optimizers.GSDR import GSDR
from core.optimizers.AGSDR import AGSDR
from systems.visualizers.compute_kappa import compute_kappa
from systems.visualizers.calculate_firing_rates import calculate_firing_rates

from jax import config
config.update("jax_platform_name", "cpu")

def run_comparison():
    num_e, num_i = 8, 2
    dt, t_max = 0.1, 1000.0
    target_afr = 10.0
    num_trials = 1000
    
    def setup_net():
        comp = jx.Compartment()
        branch = jx.Branch(comp, ncomp=1)
        e_cells = [jx.Cell(branch, parents=[-1]) for _ in range(num_e)]
        i_cells = [jx.Cell(branch, parents=[-1]) for _ in range(num_i)]
        
        for cell in e_cells + i_cells:
            cell.insert(jx.channels.HH())
            cell.insert(Inoise(initial_amp_noise=0.05, initial_mean=0.0, initial_tau=20.0))
            
        net = jx.Network(e_cells + i_cells)
        from jaxley.connect import fully_connect
        
        # Connect
        fully_connect(net.cell(range(num_e)), net.cell('all'), GradedAMPA(g=1.0))
        fully_connect(net.cell(range(num_e, num_e+num_i)), net.cell('all'), GradedGABAa(g=2.0))
        
        # Make trainable on the Network object directly
        net.make_trainable("gAMPA")
        net.make_trainable("gGABAa")
        return net

    def get_loss_fn(net):
        def loss_fn(params, key):
            # No input, just baseline noise
            net.delete_recordings()
            net.delete_stimuli()
            
            # We record all soma
            net.cell('all').branch(0).loc(0.0).record()
            
            traces = jx.integrate(net, params=params, delta_t=dt, t_max=t_max)
            
            # AFR Loss
            afr = calculate_firing_rates(traces, dt)
            afr_loss = jnp.mean(jnp.square(afr - target_afr))
            
            # Kappa Loss (target 0)
            threshold = -20.0
            spikes = (traces[:, :-1] < threshold) & (traces[:, 1:] >= threshold)
            spike_matrix = jnp.zeros_like(traces).at[:, 1:].set(spikes.astype(jnp.float32))
            kappa = compute_kappa(spike_matrix, 1000.0/dt)
            kappa_loss = jnp.abs(kappa)
            
            # Adding physical barrier check via penalty if AFR is near 0
            penalty = jnp.where(jnp.mean(afr) < 0.5, 100.0, 0.0)
            
            return afr_loss + 20.0 * kappa_loss + penalty, (afr, kappa)
        return loss_fn

    results = {}
    
    for name, opt_class in [("GSDR", GSDR), ("AGSDR", AGSDR)]:
        print(f"🚀 Training with {name}...")
        net = setup_net()
        loss_func = get_loss_fn(net)
        
        inner = optax.adam(1e-2)
        if name == "GSDR":
            optimizer = opt_class(inner, checkpoint_n=50)
        else:
            optimizer = opt_class(inner, checkpoint_n=50, ema_momentum=0.9, alpha_min=0.1)
            
        params = net.get_parameters()
        state = optimizer.init(params)
        
        history = {"loss": [], "afr": [], "kappa": [], "alpha": []}
        
        val_grad_fn = jax.jit(jax.value_and_grad(loss_func, has_aux=True))
        
        for i in range(num_trials):
            key = jax.random.PRNGKey(i + 42)
            (loss_val, (afr, kappa)), grads = val_grad_fn(params, key)
            
            updates, state = optimizer.update(grads, state, params=params, value=loss_val, key=key)
            params = optax.apply_updates(params, updates)
            
            history["loss"].append(float(loss_val))
            history["afr"].append(float(jnp.mean(afr)))
            history["kappa"].append(float(kappa))
            history["alpha"].append(float(state.a))
            
            if i % 10 == 0:
                print(f"  Trial {i} | Loss: {loss_val:.4f} | Alpha: {state.a:.4f} | Kappa: {kappa:.4f} | AFR: {jnp.mean(afr):.2f}")
        
        results[name] = history

    print("🎨 Generating Comparison Plot...")
    fig, axs = plt.subplots(2, 2, figsize=(15, 10), facecolor='white')
    
    for name, hist in results.items():
        axs[0, 0].plot(hist["loss"], label=name, alpha=0.8)
        axs[0, 1].plot(hist["alpha"], label=name, alpha=0.8)
        axs[1, 0].plot(hist["afr"], label=name, alpha=0.8)
        axs[1, 1].plot(hist["kappa"], label=name, alpha=0.8)
        
    axs[0, 0].set_title("Loss Trajectory")
    axs[0, 1].set_title("Alpha (Mixing Parameter)")
    axs[1, 0].set_title("Average Firing Rate (Target: 10Hz)")
    axs[1, 1].set_title("Kappa Synchrony (Target: 0)")
    
    for ax in axs.flatten():
        ax.legend()
        ax.grid(True, alpha=0.3)
        
    plt.tight_layout()
    save_path = "/Users/hamednejat/workspace/media/figures/optimizer_tests/baseline_10_ei_comparison.png"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300)
    print(f"✨ SUCCESS: Comparison plot saved to {save_path}")

if __name__ == "__main__":
    run_comparison()
