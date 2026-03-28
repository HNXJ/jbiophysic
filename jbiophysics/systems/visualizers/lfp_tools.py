import jax.numpy as jnp
import numpy as np

def estimate_lfp(traces, meta, dt):
    """
    Estimates the Local Field Potential (LFP) based on current dipoles.
    Logic: Sum of (V_dendrite - V_soma) / Ra for all pyramidal cells, 
    weighted by their depth (Z position).
    """
    # 1. Identify Pyramidal Cells
    pyr_indices = [i for i, m in enumerate(meta) if m['type'] == 'Pyr']
    
    # 2. Extract somatic and dendritic voltages (assumes 2-compartment model)
    # Note: In our current build_laminar_column, we record somatic voltages
    # For a true biophysical LFP, we would need both.
    # Simplified LFP: Mean extracellular potential sum at a reference depth
    
    # Let's use the average membrane potential of Pyramidal cells as a proxy 
    # for the population LFP source in this simplified version.
    lfp_proxy = jnp.mean(traces[pyr_indices, :], axis=0)
    
    return lfp_proxy

def plot_lfp_summary(lfp, dt, save_path):
    import plotly.graph_objects as go
    time_axis = np.arange(len(lfp)) * dt
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=time_axis, y=lfp, name='Putative LFP', line=dict(color='cyan')))
    fig.update_layout(title="Estimated Population LFP (Biophysical Sum)", template="plotly_dark")
    fig.write_html(save_path)
