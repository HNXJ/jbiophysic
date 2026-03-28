# -*- coding: utf-8 -*-
"""generate_review_figures
Loads fine-tuned GSDR parameters and generates validation figures for PLOS ONE review.
Focus: 40Hz Gamma and minimal Kappa (synchrony).
"""

import os
import jax
import numpy as np
import jaxley as jx
import jax.numpy as jnp
import matplotlib.pyplot as plt
import pickle
from AAE.gsdr import (
    build_net_eig, plot_full_simulation_summary,
    compute_kappa, traces_to_spike_matrix, # Assuming I need these
    noise_current_ac
)

# Configuration
from jax import config
config.update("jax_enable_x64", True)
config.update("jax_platform_name", "cpu")

# Network Hyperparameters
Ne, Nig, Nil = 36, 8, 6
dt_global = 0.1
t_max_global = 1500.0
time_axis = np.arange(0, t_max_global + dt_global, dt_global)

# Build Network
net = build_net_eig(Ne, Nig, Nil, seed=None)

# Load Fine-tuned Parameters
param_file = "/Users/hamednejat/workspace/Computational/jbiophysics/final_params_kappa_0.pkl"
if os.path.exists(param_file):
    print(f"📦 Loading parameters from {param_file}...")
    with open(param_file, "rb") as f:
        final_params = pickle.load(f)
else:
    print(f"❌ Error: {param_file} not found. Run training first.")
    exit(1)

# Run Simulation
print("🚀 Running validation simulation...")
input_amp = 0.1 # Match the training input amplitude
ac_currents = noise_current_ac(
    i_delay=500.0, i_dur=500.0, amp_n=0.0, amp_b=input_amp,
    spect=jnp.array([120.0]), delta_t=dt_global, t_max=t_max_global
)

net.delete_stimuli()
net.delete_recordings()
data_stimuli = net.cell(list(range(0, Ne, 2))).branch(1).loc(0.0).data_stimulate(ac_currents)
net.cell("all").branch(0).loc(0.0).record()

# Integration
checkpoints = [int(np.ceil((t_max_global/dt_global)**(1/2))) for _ in range(2)]
traces = np.array(jx.integrate(
    net, params=final_params, data_stimuli=data_stimuli, 
    delta_t=dt_global, t_max=t_max_global, checkpoint_lengths=checkpoints
))

# Analysis: Kappa
print("📊 Analyzing synchrony (Kappa)...")
threshold = -20.0
spikes = (traces[:, :-1] < threshold) & (traces[:, 1:] >= threshold)
spike_matrix = np.zeros_like(traces)
spike_matrix[:, 1:][spikes] = 1.0

fs = 1000.0 / dt_global
k_pre = compute_kappa(spike_matrix[:, int(100/dt_global):int(500/dt_global)], fs)
k_stim = compute_kappa(spike_matrix[:, int(500/dt_global):int(1000/dt_global)], fs)
k_post = compute_kappa(spike_matrix[:, int(1000/dt_global):int(1400/dt_global)], fs)

print(f"✅ Kappa (Pre-Stim): {k_pre:.4f}")
print(f"✅ Kappa (Stim):     {k_stim:.4f}")
print(f"✅ Kappa (Post-Stim): {k_post:.4f}")

# Plotting
print("🎨 Generating PLOS ONE summary figure...")
fig_dir = "/Users/hamednejat/workspace/media/figures/review_GSDR01"
os.makedirs(fig_dir, exist_ok=True)
savename = os.path.join(fig_dir, "review_summary_38hz_asynchronous.svg")

plot_full_simulation_summary(
    traces, time_axis, dt_global,
    title_suffix=f" (Stim: 120Hz AC, Target: 38Hz, Kappa: {k_stim:.3f})",
    save=True, savename=savename
)

print(f"✨ Figure saved to: {savename}")
