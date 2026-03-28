# -*- coding: utf-8 -*-
"""biophys_jx_gsdr
Optimized version for 50-neuron (36E-14I) network.
Target: 40Hz Gamma during 120Hz AC stimulation, minimal Kappa.
High Kappa penalty for PLOS ONE review.
"""

import os
import jax
import numpy as np
import jaxley as jx
import jax.numpy as jnp
import matplotlib.pyplot as plt
import optax
import pickle

# Foundational Imports from AAE Package
from AAE.gsdr import (
    build_net_eig, GSDR, ClampTransform, 
    get_loss_fn, train_net,
    plot_full_simulation_summary
)

# Configuration
from jax import config
config.update("jax_enable_x64", True)
config.update("jax_platform_name", "cpu")

# Paths
BASE_PATH = "/Users/hamednejat/workspace/Computational/jbiophysics"
os.makedirs(BASE_PATH, exist_ok=True)

# Network Hyperparameters: 36 Excitatory, 14 Inhibitory (8 Ing + 6 Inl)
Ne, Nig, Nil = 36, 8, 6
dt_global = 0.1
t_max_global = 1500
global_psd_interval = jnp.linspace(1.0, 100.0, 100)

# Build Network
net = build_net_eig(Ne, Nig, Nil)

# Parameter Initialization
net.edges.gAMPA = np.clip(np.random.uniform(0.0, 5.0, net.edges.gAMPA.shape), 0, 8.0)
net.edges.gGABAa = np.clip(np.random.uniform(2.0, 8.0, net.edges.gGABAa.shape), 0, 10.0)

# Transform setup (Clamp conductances)
transform = jx.ParamTransform([
    {"gAMPA": ClampTransform(0.0, 10.0)},
    {"gGABAa": ClampTransform(0.0, 10.0)}
])

# Training setup
net.delete_recordings()
net.delete_stimuli()
net.delete_trainables()
net.GradedAMPA.edge("all").make_trainable("gAMPA")
net.GradedGABAa.edge("all").make_trainable("gGABAa")

# Extract edge indices for Mutual-correlation dependent plasticity (MCDP)
ampa_pre_inds = jnp.array(net.GradedAMPA.edge("all").edges["pre_index"].values.astype(int))
ampa_post_inds = jnp.array(net.GradedAMPA.edge("all").edges["post_index"].values.astype(int))
gaba_pre_inds = jnp.array(net.GradedGABAa.edge("all").edges["pre_index"].values.astype(int))
gaba_post_inds = jnp.array(net.GradedGABAa.edge("all").edges["post_index"].values.astype(int))

# Setup Target Label: Gaussian peak at 40Hz
target_freq = 40.0
target_sigma = 5.0
target_psd = jnp.exp(-0.5 * ((global_psd_interval - target_freq) / target_sigma)**2)
target_psd = target_psd / (jnp.max(target_psd) + 1e-6)

# Dataloader
inputs = jnp.array([0.1])
labels = jnp.array([target_psd])
dataloader = [(inputs, labels)]

# Load existing optimal parameters
param_file = os.path.join(BASE_PATH, "final_params_40hz_finetuned.pkl")
if os.path.exists(param_file):
    print(f"📦 Loading existing parameters from {param_file} for intense training...")
    with open(param_file, "rb") as f:
        initial_params_finetune = pickle.load(f)
else:
    param_file = os.path.join(BASE_PATH, "optimal_params_40hz.pkl")
    if os.path.exists(param_file):
        print(f"📦 Loading existing parameters from {param_file} for intense training...")
        with open(param_file, "rb") as f:
            initial_params_finetune = pickle.load(f)
    else:
        print("⚠️ No existing parameter file found. Starting from scratch.")
        initial_params_finetune = None

# Setup Loss and Optimizer with much higher Kappa weight
checkpoints = [int(np.ceil((t_max_global/dt_global)**(1/2))) for _ in range(2)]
loss_fn = get_loss_fn(net, transform, dt_global, global_psd_interval, 
                      lower_c=5.0, upper_c=50.0, firing_rate_weight=10.0, psd_weight=1000.0,
                      num_e=Ne, checkpoints=checkpoints, kappa_weight=10000.0)

optimizer_inner = optax.adam(learning_rate=1e-2)
optimizer = GSDR(
    inner_optimizer=optimizer_inner,
    a_init=0.5,
    a_dynamic=True,
    checkpoint_n=10,
    mcdp=True
)

# Run Training
print(f"🚀 Training with Intense Kappa Minimization (Penalty: 10000.0)...")
print(f"🎯 Target: 40Hz Gamma | Minimize Kappa | LR: 0.01")

final_params, training_log = train_net(
    net, optimizer, transform, dataloader, loss_fn,
    ampa_pre_inds, ampa_post_inds, gaba_pre_inds, gaba_post_inds,
    dt_global, band_definitions={}, epoch_n=100,
    initial_params=initial_params_finetune
)

print("✅ Training complete. Final Loss:", training_log["loss"][-1])

# Save final results
save_path = os.path.join(BASE_PATH, "final_params_low_kappa.pkl")
with open(save_path, "wb") as f:
    pickle.dump(final_params, f)
print(f"✨ Parameters saved to: {save_path}")
