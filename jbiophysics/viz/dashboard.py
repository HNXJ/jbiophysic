"""Combined multi-panel Plotly dashboard for jbiophysics."""
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import signal

def generate_dashboard(report, title="Jbiophysics — Simulation Dashboard") -> go.Figure:
    has_loss = len(report.loss_history) > 0
    n_rows = 4 if has_loss else 3
    subtitles = ["Spike Raster", "Time-Frequency", "Power Spectral Density"]
    if has_loss: subtitles.append("Optimization Loss")
    
    fig = make_subplots(
        rows=n_rows, cols=1, 
        subplot_titles=subtitles, 
        vertical_spacing=0.06,
        row_heights=[0.3, 0.3, 0.25] + ([0.15] if has_loss else [])
    )
    
    traces_data = report.traces
    dt = report.dt
    num_cells, num_steps = traces_data.shape
    time_ms = np.arange(num_steps) * dt
    threshold = -20.0

    # Panel 1: Raster
    spike_times, spike_neurons = [], []
    for i in range(num_cells):
        crossings = (traces_data[i, :-1] < threshold) & (traces_data[i, 1:] >= threshold)
        indices = np.where(crossings)[0]
        spike_times.extend(time_ms[indices])
        spike_neurons.extend([i]*len(indices))
    
    fig.add_trace(go.Scattergl(
        x=spike_times, y=spike_neurons, mode="markers",
        marker=dict(size=2, color="#4FC3F7", opacity=0.8), 
        showlegend=False
    ), row=1, col=1)
    fig.update_yaxes(title_text="Neuron", row=1, col=1)

    # Panel 2: Spectrogram (Layer Average)
    try:
        from scipy.ndimage import gaussian_filter1d
        fs = 1000.0/dt
        mean_v = np.mean(traces_data, axis=0)
        f_s, t_s, Sxx = signal.spectrogram(mean_v - np.mean(mean_v), fs=fs, nperseg=int(fs*0.25))
        mask = (f_s >= 1) & (f_s <= 100)
        fig.add_trace(go.Heatmap(
            z=np.log10(Sxx[mask, :] + 1e-12), x=t_s*1000, y=f_s[mask],
            colorscale="Viridis", showscale=False
        ), row=2, col=1)
    except: pass
    fig.update_yaxes(title_text="Freq (Hz)", row=2, col=1)

    # Panel 3: PSD
    mean_v = np.mean(traces_data, axis=0)
    f_p, pxx = signal.welch(mean_v - np.mean(mean_v), fs=1000.0/dt, nperseg=int(1000.0/dt))
    mask = (f_p >= 1) & (f_p <= 100)
    fig.add_trace(go.Scatter(
        x=f_p[mask], y=pxx[mask]/np.max(pxx[mask]), mode="lines",
        line=dict(color="#CFB87C", width=2), fill="tozeroy", showlegend=False
    ), row=3, col=1)
    fig.update_yaxes(title_text="Norm Power", row=3, col=1)

    if has_loss:
        fig.add_trace(go.Scatter(
            x=list(range(len(report.loss_history))), y=report.loss_history,
            mode="lines", line=dict(color="#9400D3", width=2), showlegend=False
        ), row=4, col=1)
        fig.update_yaxes(title_text="Loss", row=4, col=1)

    fig.update_layout(template="plotly_dark", height=250*n_rows, title=title)
    return fig

def generate_laminar_dashboard(report, title="Spectrolaminar Motif Analysis") -> go.Figure:
    """
    Generates the specific dashboard requested by the user.
    - 4 rasters (by layer)
    - 4 relative power plots
    - LFP/PSD
    - 3D Schematic
    """
    offsets = report.metadata.get("population_offsets", {})
    if not offsets:
        return generate_dashboard(report, title=title + " (Fallback)")

    layers = ["L23", "L4", "L56"]
    n_layers = len(layers)
    
    # Grid: 
    # Left Column (Rasters): 1, 2, 3
    # Mid Column (Power): 1, 2, 3
    # Bottom Row: PSD (Wide)
    # Right Column: 3D Schematic (Tall)
    
    fig = make_subplots(
        rows=4, cols=3,
        specs=[[{}, {}, {"rowspan": 3, "type": "scene"}],
               [{}, {}, None],
               [{}, {}, None],
               [{"colspan": 2}, None, {}]],
        subplot_titles=[
            "L2/3 Raster", "L2/3 Rel. Power", "3D Network Schematic",
            "L4 Raster", "L4 Rel. Power",
            "L5/6 Raster", "L5/6 Rel. Power",
            "Global LFP PSD", "Layer Metrics"
        ],
        vertical_spacing=0.08,
        horizontal_spacing=0.08
    )

    dt = report.dt
    traces = report.traces
    threshold = -20.0
    time_ms = np.arange(traces.shape[1]) * dt
    fs = 1000.0/dt

    # Color Palette
    colors = {"L23": "#00FFFF", "L4": "#CFB87C", "L56": "#9400D3"}

    for i, layer in enumerate(layers):
        # Find neurons in this layer
        start, end = -1, -1
        for pop, (s, e) in offsets.items():
            if layer in pop:
                if start == -1 or s < start: start = s
                if end == -1 or e > end: end = e
        
        if start == -1: continue

        # --- Raster ---
        layer_traces = traces[start:end, :]
        spike_times, spike_neurons = [], []
        for n_idx in range(layer_traces.shape[0]):
            crossings = (layer_traces[n_idx, :-1] < threshold) & (layer_traces[n_idx, 1:] >= threshold)
            idx = np.where(crossings)[0]
            spike_times.extend(time_ms[idx])
            spike_neurons.extend([n_idx + start]*len(idx))
        
        fig.add_trace(go.Scattergl(
            x=spike_times, y=spike_neurons, mode="markers",
            marker=dict(size=2, color=colors[layer]), showlegend=False
        ), row=i+1, col=1)

        # --- Relative Power ---
        lfp = np.mean(layer_traces, axis=0)
        f, pxx = signal.welch(lfp - np.mean(lfp), fs=fs, nperseg=int(fs))
        gamma_mask = (f >= 40) & (f <= 90)
        beta_mask = (f >= 15) & (f <= 25)
        
        # We'll plot a bar chart of Gamma vs Beta for this layer
        rel_g = np.mean(pxx[gamma_mask])
        rel_b = np.mean(pxx[beta_mask])
        total = rel_g + rel_b + 1e-12
        
        fig.add_trace(go.Bar(
            x=["Gamma", "Beta"], y=[rel_g/total, rel_b/total],
            marker_color=[colors["L23"], colors["L56"]], showlegend=False
        ), row=i+1, col=2)

    # --- Global PSD ---
    global_lfp = np.mean(traces, axis=0)
    f_g, pxx_g = signal.welch(global_lfp - np.mean(global_lfp), fs=fs, nperseg=int(fs))
    mask_g = (f_g >= 1) & (f_g <= 100)
    fig.add_trace(go.Scatter(
        x=f_g[mask_g], y=pxx_g[mask_g], mode="lines",
        line=dict(color="#FFFFFF", width=3), fill="tozeroy", showlegend=False
    ), row=4, col=1)

    # --- 3D Schematic ---
    # We mock neuron positions based on layer if not available
    # L2/3: Z=[200, 300], L4: Z=[100, 200], L5/6: Z=[0, 100]
    # In a real scenario, builder.xyz would be here.
    n_total = traces.shape[0]
    x = np.random.uniform(-50, 50, n_total)
    y = np.random.uniform(-50, 50, n_total)
    z = np.zeros(n_total)
    c_list = []
    for pop, (s, e) in offsets.items():
        if "L23" in pop: z[s:e] = np.random.uniform(200, 300, e-s); c_list.extend([colors["L23"]]*(e-s))
        elif "L4" in pop: z[s:e] = np.random.uniform(100, 200, e-s); c_list.extend([colors["L4"]]*(e-s))
        elif "L56" in pop: z[s:e] = np.random.uniform(0, 100, e-s); c_list.extend([colors["L56"]]*(e-s))
        else: c_list.extend(["#FFFFFF"]*(e-s))

    fig.add_trace(go.Scatter3d(
        x=x, y=y, z=z, mode="markers",
        marker=dict(size=3, color=c_list, opacity=0.7), showlegend=False
    ), row=1, col=3)

    fig.update_layout(
        template="plotly_dark", 
        height=1000, 
        title=dict(text=title, font=dict(size=24, color="#CFB87C")),
        margin=dict(l=50, r=50, t=100, b=50)
    )
    fig.update_xaxes(title_text="Time (ms)", row=3, col=1)
    fig.update_xaxes(title_text="Freq (Hz)", row=4, col=1)
    
    return fig
