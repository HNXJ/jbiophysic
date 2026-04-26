# src/jbiophysic/viz/plotly/tracer_plots.py
import plotly.graph_objects as go
import numpy as np

def plot_tracer_voltage(times: np.ndarray, voltages: np.ndarray):
    """Generates a Plotly figure for the tracer bullet voltage trace."""
    print("Generating Tracer Bullet Plotly figure")
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=times, 
        y=voltages, 
        mode='lines', 
        name='Membrane Potential',
        line=dict(color='#CFB87C', width=2)
    ))
    
    fig.update_layout(
        title="Tracer Bullet: Leaky Integrate-and-Fire Dynamics",
        xaxis_title="Time (ms)",
        yaxis_title="Voltage (mV)",
        template="plotly_dark",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#CFB87C')
    )
    
    print("Tracer plot generation complete.")
    return fig
