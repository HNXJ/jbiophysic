# -*- coding: utf-8 -*-
"""biophys_jx_gsdr_long
Continuous GSDR training loop (up to 1000 trials).
Target: 38Hz Gamma during stim, 1/f during off-stim, Kappa ≈ 0 throughout.
"""

import os
import jax
import numpy as np
import jaxley as jx
import jax.numpy as jnp
import matplotlib.pyplot as plt
import optax
import pickle

from AAE.gsdr import (
    build_net_eig, GSDR, ClampTransform, 
    get_loss_fn, train_net, compute_kappa
)

from jax import config
config.update("jax_enable_x64", True)
config.update("jax_platform_name", "cpu")

BASE_PATH = "/Users/hamednejat/workspace/Computational/jbiophysics"
os.makedirs(BASE_PATH, exist_ok=True)

Ne, Nig, Nil = 36, 8, 6
dt_global = 0.1
t_max_global = 1500
global_psd_interval = jnp.linspace(1.0, 100.0, 100)

# 1. Targets
target_stim_freq = 38.0
target_sigma = 5.0
target_psd_stim = jnp.exp(-0.5 * ((global_psd_interval - target_stim_freq) / target_sigma)**2)
target_psd_stim = target_psd_stim / (jnp.max(target_psd_stim) + 1e-6)

target_psd_off = 1.0 / (global_psd_interval + 1e-6)
target_psd_off = target_psd_off / (jnp.max(target_psd_off) + 1e-6)

inputs = jnp.array([0.1])
labels = jnp.array([[target_psd_stim, target_psd_off]]) # Shape (1, 2, 100)
dataloader = [(inputs, labels)]

# 2. Network & Transform
net = build_net_eig(Ne, Nig, Nil)
transform = jx.ParamTransform([
    {"gAMPA": ClampTransform(0.0, 10.0)},
    {"gGABAa": ClampTransform(0.0, 10.0)}
])

net.delete_recordings()
net.delete_stimuli()
net.delete_trainables()
net.GradedAMPA.edge("all").make_trainable("gAMPA")
net.GradedGABAa.edge("all").make_trainable("gGABAa")

ampa_pre_inds = jnp.array(net.GradedAMPA.edge("all").edges["pre_index"].values.astype(int))
ampa_post_inds = jnp.array(net.GradedAMPA.edge("all").edges["post_index"].values.astype(int))
gaba_pre_inds = jnp.array(net.GradedGABAa.edge("all").edges["pre_index"].values.astype(int))
gaba_post_inds = jnp.array(net.GradedGABAa.edge("all").edges["post_index"].values.astype(int))

# 3. Optimization Setup
checkpoints = [int(np.ceil((t_max_global/dt_global)**(1/2))) for _ in range(2)]
loss_fn = get_loss_fn(net, transform, dt_global, global_psd_interval, 
                      lower_c=5.0, upper_c=50.0, firing_rate_weight=10.0, psd_weight=1000.0,
                      num_e=Ne, checkpoints=checkpoints, kappa_weight=10000.0)

optimizer_inner = optax.adam(learning_rate=1e-2)
optimizer = GSDR(
    inner_optimizer=optimizer_inner,
    a_init=0.01,
    a_dynamic=True,
    checkpoint_n=10,
    mcdp=True
)

# 4. State Management
param_file = os.path.join(BASE_PATH, "final_params_kappa_0.pkl")
if os.path.exists(param_file):
    print(f"📦 Loading {param_file}...")
    with open(param_file, "rb") as f:
        current_params = pickle.load(f)
else:
    print("⚠️ Starting from scratch...")
    net.edges.gAMPA = np.clip(np.random.uniform(0.0, 5.0, net.edges.gAMPA.shape), 0, 8.0)
    net.edges.gGABAa = np.clip(np.random.uniform(2.0, 8.0, net.edges.gGABAa.shape), 0, 10.0)
    current_params = None

# 5. Continuous Loop
max_chunks = 10
trials_per_chunk = 100

for chunk in range(max_chunks):
    print(f"\n--- 🚀 CHUNK {chunk + 1}/{max_chunks} ({trials_per_chunk} trials) ---")
    current_params, training_log = train_net(
        net, optimizer, transform, dataloader, loss_fn,
        ampa_pre_inds, ampa_post_inds, gaba_pre_inds, gaba_post_inds,
        dt_global, band_definitions={}, epoch_n=trials_per_chunk,
        initial_params=current_params
    )
    
    # Save checkpoint
    with open(param_file, "wb") as f:
        pickle.dump(current_params, f)
        
    # Evaluate Kappa
    from AAE.gsdr.simulation import noise_current_ac
    ac_currents = noise_current_ac(
        i_delay=500.0, i_dur=500.0, amp_n=0.0, amp_b=0.1,
        spect=jnp.array([120.0]), delta_t=dt_global, t_max=t_max_global
    )
    net.delete_stimuli()
    net.delete_recordings()
    data_stimuli = net.cell(list(range(0, Ne, 2))).branch(1).loc(0.0).data_stimulate(ac_currents)
    net.cell("all").branch(0).loc(0.0).record()
    
    traces = np.array(jx.integrate(
        net, params=current_params, data_stimuli=data_stimuli, 
        delta_t=dt_global, t_max=t_max_global, checkpoint_lengths=checkpoints
    ))
    
    threshold = -20.0
    spikes = (traces[:, :-1] < threshold) & (traces[:, 1:] >= threshold)
    spike_matrix = np.zeros_like(traces)
    spike_matrix[:, 1:][spikes] = 1.0
    
    fs = 1000.0 / dt_global
    k_pre = compute_kappa(spike_matrix[:, :int(500/dt_global)], fs)
    k_stim = compute_kappa(spike_matrix[:, int(500/dt_global):int(1000/dt_global)], fs)
    k_post = compute_kappa(spike_matrix[:, int(1000/dt_global):], fs)
    
    total_kappa = np.abs(k_pre) + np.abs(k_stim) + np.abs(k_post)
    print(f"📊 Chunk {chunk+1} Final Kappas - Pre: {k_pre:.4f} | Stim: {k_stim:.4f} | Post: {k_post:.4f}")
    
    if total_kappa <= 0.1:
        print("✅ SUCCESS: Target Kappa <= 0.1 reached!")
        break
    else:
        print(f"⚠️ Total Kappa ({total_kappa:.4f}) > 0.1. Continuing training...")

print("✨ Long training loop complete.")
