import jax
import jax.numpy as jnp
import jaxley as jx
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import os
import sys
import optax

# Add jbiophysics to path
sys.path.insert(0, os.getcwd())

from compose import NetBuilder
from neurons.cortical import Inoise, GradedAMPA, GradedGABAa, GradedGABAb, SafeHH
from optimizers.optimizers import compute_kappa, AGSDR

def run_laminar_motif_optimized():
    print("🚀 Replicating Spectrolaminar Motif (AGSDR Optimized v2)...")
    
    # 1. Builder Setup
    builder = NetBuilder(seed=42)
    
    # Populations
    builder.add_population("L23_Pyr", n=40, cell_type="pyramidal", noise_amp=0.15, noise_tau=2.0)
    builder.add_population("L23_PV", n=10, cell_type="pv", noise_amp=0.1, noise_tau=2.0)
    builder.add_population("L4_Pyr", n=20, cell_type="pyramidal", noise_amp=0.12, noise_tau=10.0)
    builder.add_population("L4_PV", n=5, cell_type="pv", noise_amp=0.1, noise_tau=10.0)
    builder.add_population("L56_Pyr", n=40, cell_type="pyramidal", noise_amp=0.15, noise_tau=50.0)
    builder.add_population("L56_PV", n=10, cell_type="pv", noise_amp=0.1, noise_tau=50.0)
    builder.add_population("L56_SST", n=10, cell_type="sst", noise_amp=0.1, noise_tau=50.0)

    # 2. Connections
    builder.connect("L23_Pyr", "L23_PV", "AMPA", p=0.4, g=2.0, tauD_AMPA=2.0)
    builder.connect("L23_PV", "L23_Pyr", "GABAa", p=0.6, g=4.0, tauD_GABAa=3.0)
    builder.connect("L4_Pyr", "L4_PV", "AMPA", p=0.4, g=2.0)
    builder.connect("L4_PV", "L4_Pyr", "GABAa", p=0.6, g=4.0)
    builder.connect("L56_Pyr", "L56_Pyr", "AMPA", p=0.3, g=3.0, tauD_AMPA=10.0)
    builder.connect("L56_PV", "L56_Pyr", "GABAa", p=0.6, g=5.0, tauD_GABAa=25.0)
    builder.connect("L56_SST", "L56_Pyr", "GABAb", p=0.5, g=3.0, tauD_GABAb=200.0)
    builder.connect("L4_Pyr", "L23_Pyr", "AMPA", p=0.2, g=1.5)
    builder.connect("L23_Pyr", "L56_Pyr", "AMPA", p=0.1, g=1.0)

    # 3. Training Setup (AGSDR)
    builder.make_trainable(["gAMPA", "gGABAa", "gGABAb"])
    net = builder.build()
    
    # 4. Optimization Loop
    dt = 0.1
    t_max = 1000.0 # Slightly longer segments for better FR estimate
    epochs = 50
    lr = 2e-2
    
    net.delete_recordings()
    net.cell("all").branch(0).loc(0.0).record("v")
    
    inner_opt = optax.adam(lr)
    optimizer = AGSDR(inner_optimizer=inner_opt, a_init=0.5)
    
    params = net.get_parameters()
    opt_state = optimizer.init(params)
    key = jax.random.PRNGKey(42)
    
    print(f"⌛ Tuning conductances via AGSDR ({epochs} epochs)...")
    
    def loss_fn(p, key):
        traces = jx.integrate(net, params=p, delta_t=dt, t_max=t_max)
        traces = jnp.nan_to_num(traces, nan=0.0)
        
        # FR Target (10 Hz)
        threshold = -20.0
        spikes = (traces[:, :-1] < threshold) & (traces[:, 1:] >= threshold)
        frs = jnp.sum(spikes, axis=1) / (t_max / 1000.0)
        
        # Smooth FR Loss
        fr_loss = jnp.mean(jnp.square(frs - 10.0))
        
        # Approximate Synchrony Loss (Variance of population mean Vm)
        # Low variance of mean = high asynchrony
        pop_mean = jnp.mean(traces, axis=0)
        sync_loss = jnp.var(pop_mean) * 100.0
        
        return fr_loss + sync_loss, traces

    for epoch in range(epochs):
        key, step_key = jax.random.split(key)
        loss, traces = loss_fn(params, step_key)
        
        # Dummy "grads" for exploration
        grads = jax.tree.map(jnp.zeros_like, params)
        
        updates, opt_state = optimizer.update(grads, opt_state, params=params, value=loss, key=step_key)
        params = optax.apply_updates(params, updates)
        
        if epoch % 10 == 0:
            mean_fr = jnp.mean(jnp.sum((traces[:,:-1] < -20) & (traces[:,1:] >= -20), 1) / (t_max/1000.0))
            print(f"  Epoch {epoch:2} | Loss: {loss:8.2f} | Mean FR: {mean_fr:5.1f} Hz")

    # 5. Final Simulation
    print("\n✅ Tuning complete. Running final 3s simulation...")
    t_max_final = 3000.0
    net.delete_recordings()
    net.cell("all").branch(0).loc(0.0).record("v")
    
    traces_final = jx.integrate(net, params=params, delta_t=dt, t_max=t_max_final)
    traces_np = np.array(traces_final)
    traces_np = np.nan_to_num(traces_np, nan=0.0)
    traces_np = np.clip(traces_np, -100.0, 100.0)

    # 6. Analysis
    threshold = -20.0
    spikes = (traces_np[:, :-1] < threshold) & (traces_np[:, 1:] >= threshold)
    frs = np.sum(spikes, axis=1) / (t_max_final / 1000.0)
    
    offsets = builder.population_offsets
    layers = ["L23", "L4", "L56"]
    
    print("\n--- Final Physiological Audit ---")
    for l in layers:
        start, end = offsets[l + "_Pyr"]
        k = compute_kappa(spikes[start:end, :], fs=1000.0/dt)
        print(f"Layer {l:5} | FR: {np.mean(frs[start:end]):5.1f} Hz | Kappa: {k:6.3f}")

    # 7. Export Results
    from export import ResultsReport
    report = ResultsReport(
        traces=traces_np,
        dt=dt,
        t_max=t_max_final,
        metadata={
            "population_offsets": builder.population_offsets,
            "method": "AGSDR",
            "title": "Spectrolaminar Motif (Replica: Bastos 2012)"
        }
    )
    report.save_json("laminar_results.json")
    print("✅ Results exported to 'laminar_results.json'")

    # 8. Motif Plot
    fs = 1000.0 / dt
    psds = []
    for l in layers:
        start, end = offsets[l + "_Pyr"]
        lfp = np.mean(traces_np[start:end, :], axis=0)
        f, pxx = signal.welch(lfp - np.mean(lfp), fs=fs, nperseg=int(fs))
        psds.append(pxx)
    psd_data = np.array(psds)

    gamma_mask = (f >= 40) & (f <= 90)
    beta_mask = (f >= 15) & (f <= 25)
    lg = [np.mean(psd_data[i, gamma_mask]) for i in range(3)]
    lb = [np.mean(psd_data[i, beta_mask]) for i in range(3)]
    
    norm_gamma = np.array(lg) / (np.max(lg) + 1e-15)
    norm_beta = np.array(lb) / (np.max(lb) + 1e-15)

    plt.figure(figsize=(6, 8))
    plt.plot(norm_gamma, [1, 2, 3], 'o-', color='#00FFFF', label='$\gamma$ (40-90 Hz)', lw=6)
    plt.plot(norm_beta, [1, 2, 3], 'o-', color='#9400D3', label='$\\beta$ (15-25 Hz)', lw=6)
    plt.yticks([1, 2, 3], ["Superficial (L2/3)", "Granular (L4)", "Deep (L5/6)"])
    plt.gca().invert_yaxis()
    plt.xlabel("Relative Spectral Power")
    plt.title("Spectrolaminar Motif: AGSDR Optimized\n(Replica: Bastos et al. 2012)")
    plt.grid(True, alpha=0.2)
    plt.legend()
    plt.tight_layout()
    plt.savefig("spectrolaminar_motif_agsdr.png", dpi=200)
    print("✅ Result saved to 'spectrolaminar_motif_agsdr.png'")

if __name__ == "__main__":
    run_laminar_motif_optimized()
