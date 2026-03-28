import os
import sys
import jax
import jax.numpy as jnp
import jaxley as jx
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
from typing import Any, List, Dict, Optional, Tuple
from scipy import signal

# --- Path Setup ---
sys.path.insert(0, '/Users/hamednejat/workspace/Repositories/AAE')
sys.path.insert(0, '/Users/hamednejat/workspace/Repositories/jbiophys')

from systems.visualizers.plot_full_simulation_summary import plot_full_simulation_summary
from systems.visualizers.spatial_3d import plot_network_3d
from systems.visualizers.lfp_tools import estimate_lfp
from systems.visualizers.calculate_firing_rates import calculate_firing_rates

def generate_plotly_summary(voltages, time_axis, dt, output_path, spike_threshold=-20.0):
    """Generates a Plotly-based HTML summary (Raster + Spectrogram)."""
    num_neurons = voltages.shape[0]
    
    spike_indices = []
    spike_times = []
    for i in range(num_neurons):
        spikes = (voltages[i, :-1] < spike_threshold) & (voltages[i, 1:] >= spike_threshold)
        times = time_axis[1:][spikes]
        spike_times.extend(times.tolist())
        spike_indices.extend([i] * len(times))

    fs = 1000.0 / dt
    mean_v = np.mean(voltages, axis=0)
    freqs, times, Sxx = signal.spectrogram(mean_v, fs=fs, nperseg=int(250/dt))
    
    fig = make_subplots(rows=2, cols=1, subplot_titles=("Raster Plot", "Mean Spectrogram"), vertical_spacing=0.1)
    fig.add_trace(go.Scatter(x=spike_times, y=spike_indices, mode='markers', marker=dict(size=3, color='white'), name="Spikes"), row=1, col=1)
    fig.add_trace(go.Heatmap(x=times*1000, y=freqs, z=np.log10(Sxx + 1e-9), colorscale='Jet', name="Power"), row=2, col=1)
    fig.update_yaxes(title_text="Frequency (Hz)", range=[1, 100], row=2, col=1)
    fig.update_layout(height=1000, width=1000, template="plotly_dark")
    fig.write_html(output_path)

def plot_training_history(log: Dict, output_path: str):
    """Generates an optimization trajectory report."""
    epochs = range(len(log["loss"]))
    fig = make_subplots(rows=4, cols=1, subplot_titles=("Loss Trajectory", "Energy Consumption", "Global Firing Rate", "Parameter Change"), vertical_spacing=0.08)
    
    fig.add_trace(go.Scatter(x=list(epochs), y=log["loss"], name="Loss", line=dict(color='gold')), row=1, col=1)
    fig.add_trace(go.Scatter(x=list(epochs), y=log["energy"], name="Energy", line=dict(color='magenta')), row=2, col=1)
    fig.add_trace(go.Scatter(x=list(epochs), y=log["firing_rate"], name="Global FR", line=dict(color='cyan')), row=3, col=1)
    fig.add_trace(go.Scatter(x=list(epochs), y=log["param_change"], name="dParam/P", line=dict(color='white')), row=4, col=1)
    
    fig.update_layout(height=1200, width=1000, template="plotly_dark", title_text="Optimization History")
    fig.write_html(output_path)

def run_standardized_visualizer(
    net: jx.Network,
    params_list: List[Any],
    labels: List[str],
    meta: List[Dict],
    training_log: Optional[Dict] = None,
    name_label: str = "integrated_run",
    t_max: float = 1500.0,
    dt: float = 0.1
):
    """Standardized Visualizer Pipeline: Supports multiple snapshots and memory-efficient plotting."""
    base_dir = f"/Users/hamednejat/workspace/Computational/mscz/figures/{name_label}"
    os.makedirs(base_dir, exist_ok=True)
    print(f"🎬 Standardized Visualizer: Generating snapshots in {base_dir}...")

    # 1. Plot Training History if log is provided
    if training_log:
        history_path = os.path.join(base_dir, "optimization_history.html")
        plot_training_history(training_log, history_path)

    # 2. Iterate through Parameter Snapshots
    for params, label in zip(params_list, labels):
        snapshot_dir = os.path.join(base_dir, label)
        os.makedirs(snapshot_dir, exist_ok=True)
        print(f"   🖼️ Snapshot {label}...")

        # Memory Efficient Simulation for Visualization
        # We simulate in 1s chunks to avoid giant allocations
        chunk_ms = 1000.0
        num_chunks = int(t_max / chunk_ms)
        decimation = 10 # 0.1ms -> 1ms for plotting
        
        all_decimated_voltages = []
        all_spike_times = []
        all_spike_indices = []
        
        # Note: JAXley integrate doesn't easily support state carry-over in a simple loop
        # So we simulate the full block but immediately decimate to save RAM
        # If t_max is very large, we apply jax.jit to the integration to optimize
        
        voltages = jx.integrate(net, params=params, delta_t=dt, t_max=t_max)
        
        # Calculate spike data at full resolution
        num_neurons = voltages.shape[0]
        time_axis_full = np.arange(voltages.shape[1]) * dt
        threshold = -20.0
        
        for i in range(num_neurons):
            spikes = (voltages[i, :-1] < threshold) & (voltages[i, 1:] >= threshold)
            times = time_axis_full[1:][spikes]
            all_spike_times.extend(times.tolist())
            all_spike_indices.extend([i] * len(times))
            
        # Decimate voltages for plotting (10x reduction)
        voltages_decimated = voltages[:, ::decimation]
        time_axis_decimated = time_axis_full[::decimation]

        # --- Interactive Reports ---
        
        # Plotly Dynamics Summary (using decimated data and collected spikes)
        print("      📊 Generating Plotly Dynamics Summary...")
        fig_dyn = make_subplots(rows=2, cols=1, subplot_titles=("Raster Plot", "Mean Spectrogram"), vertical_spacing=0.1)
        fig_dyn.add_trace(go.Scatter(x=all_spike_times, y=all_spike_indices, mode='markers', marker=dict(size=3, color='white'), name="Spikes"), row=1, col=1)
        
        # Spectrogram on decimated mean
        fs_dec = 1000.0 / (dt * decimation)
        mean_v_dec = np.mean(voltages_decimated, axis=0)
        freqs, times_spec, Sxx = signal.spectrogram(mean_v_dec, fs=fs_dec, nperseg=int(250/(dt*decimation)))
        fig_dyn.add_trace(go.Heatmap(x=times_spec*1000, y=freqs, z=np.log10(Sxx + 1e-9), colorscale='Jet', name="Power"), row=2, col=1)
        fig_dyn.update_yaxes(title_text="Frequency (Hz)", range=[1, 100], row=2, col=1)
        fig_dyn.update_layout(height=1000, width=1000, template="plotly_dark")
        fig_dyn.write_html(os.path.join(snapshot_dir, "dynamics.html"))

        plot_network_3d(net, meta, os.path.join(snapshot_dir, "architecture.html"))
        
        # Biophysical Suite
        fig_suite = make_subplots(rows=3, cols=1, subplot_titles=("Population Avg Vm", "LFP Proxy", "FR Distribution"))
        fig_suite.add_trace(go.Scatter(x=time_axis_decimated, y=np.mean(voltages_decimated, axis=0), name="Avg Vm"), row=1, col=1)
        
        # LFP (estimated on decimated for speed)
        lfp = estimate_lfp(voltages_decimated, meta, dt * decimation)
        fig_suite.add_trace(go.Scatter(x=time_axis_decimated, y=lfp, name="LFP"), row=2, col=1)
        
        # FR Distribution (calculated from full spikes for accuracy)
        spike_counts = np.zeros(num_neurons)
        for idx in all_spike_indices:
            spike_counts[idx] += 1
        frs = spike_counts / (t_max / 1000.0)
        fig_suite.add_trace(go.Histogram(x=frs, name="FRs"), row=3, col=1)
        
        fig_suite.update_layout(height=1000, width=1000, template="plotly_dark")
        fig_suite.write_html(os.path.join(snapshot_dir, "biophysical_suite.html"))

    print(f"✅ Visualization Pipeline Complete.")
