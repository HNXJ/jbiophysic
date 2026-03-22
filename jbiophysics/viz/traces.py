"""Plotly voltage trace viewer."""
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def plot_traces(report, cell_indices=None, max_cells=10, title="Voltage Traces") -> go.Figure:
    traces = report.traces; dt = report.dt; num_cells = traces.shape[0]
    if cell_indices is None: cell_indices = list(range(min(max_cells, num_cells)))
    time_ms = np.arange(traces.shape[1]) * dt
    fig = make_subplots(rows=len(cell_indices), cols=1, shared_xaxes=True, vertical_spacing=0.02)
    colors = ["#4FC3F7","#81C784","#FFB74D","#E57373","#BA68C8","#4DB6AC","#FFD54F","#7986CB","#A1887F","#90A4AE"]
    for i, idx in enumerate(cell_indices):
        fig.add_trace(go.Scattergl(x=time_ms, y=traces[idx], mode="lines",
            line=dict(color=colors[i%len(colors)], width=1), name=f"Cell {idx}"), row=i+1, col=1)
        fig.update_yaxes(showticklabels=False, row=i+1, col=1)
    fig.update_layout(title=dict(text=title, font=dict(size=16, family="Inter")),
        template="plotly_dark", height=80*len(cell_indices)+100,
        margin=dict(l=40, r=20, t=50, b=50), showlegend=False)
    fig.update_xaxes(title_text="Time (ms)", row=len(cell_indices), col=1)
    return fig
