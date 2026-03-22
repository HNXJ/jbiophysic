"""Plotly raster plot from voltage traces."""
import numpy as np
import plotly.graph_objects as go

def plot_raster(report, threshold: float = -20.0, title: str = "Spike Raster") -> go.Figure:
    traces = report.traces
    dt = report.dt
    num_cells, num_steps = traces.shape
    time_ms = np.arange(num_steps) * dt
    spike_times, spike_neurons = [], []
    for i in range(num_cells):
        crossings = (traces[i, :-1] < threshold) & (traces[i, 1:] >= threshold)
        indices = np.where(crossings)[0]
        spike_times.extend(time_ms[indices])
        spike_neurons.extend([i] * len(indices))
    fig = go.Figure()
    fig.add_trace(go.Scattergl(x=spike_times, y=spike_neurons, mode="markers",
        marker=dict(size=2, color="#4FC3F7", opacity=0.8), name="Spikes"))
    fig.update_layout(title=dict(text=title, font=dict(size=16, family="Inter")),
        xaxis_title="Time (ms)", yaxis_title="Neuron Index", template="plotly_dark",
        height=400, margin=dict(l=60, r=20, t=50, b=50), yaxis=dict(range=[-0.5, num_cells-0.5]))
    return fig
