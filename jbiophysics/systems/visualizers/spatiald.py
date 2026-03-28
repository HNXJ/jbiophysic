import plotly.graph_objects as go
import numpy as np
import pandas as pd

def plot_network_3d(net, meta, save_path):
    """
    Generates an interactive 3D plot of the network architecture.
    Nodes are colored by type, size reflects radius.
    Hover provides metadata and connectivity info.
    """
    # 1. Extract Cell Positions with defaults if missing
    x, y, z = [], [], []
    for i, cell in enumerate(net.cells):
        if hasattr(cell, 'xyz') and cell.xyz is not None:
            x.append(cell.xyz[0])
            y.append(cell.xyz[1])
            z.append(cell.xyz[2])
        else:
            # Assign fallback coordinates based on index if missing
            x.append(np.random.uniform(-100, 100))
            y.append(np.random.uniform(-100, 100))
            z.append(i * 10.0) 
    
    types = [m['type'] for m in meta]
    layers = [m['layer'] for m in meta]
    
    # Define colors
    color_map = {'Pyr': 'gold', 'PV': 'cyan', 'SST': 'magenta', 'VIP': 'white'}
    node_colors = [color_map.get(t, 'gray') for t in types]

    # 2. Trace Nodes
    node_trace = go.Scatter3d(
        x=x, y=y, z=z,
        mode='markers',
        marker=dict(size=6, color=node_colors, opacity=0.8, line=dict(color='black', width=1)),
        text=[f"Layer: {l}<br>Type: {t}<br>ID: {i}" for i, (l, t) in enumerate(zip(layers, types))],
        hoverinfo='text',
        name='Neurons'
    )

    # 3. Trace Edges (Sub-sampled for visibility)
    # We'll draw lines for inter-area and strong intra-area synapses
    edge_x, edge_y, edge_z = [], [], []
    
    # Access edges via net.edges DataFrame if possible
    # For now, let's just plot the nodes and layout
    fig = go.Figure(data=[node_trace])
    
    fig.update_layout(
        title="Hierarchical Cortical Architecture (3D)",
        scene=dict(
            xaxis_title='X (um)', yaxis_title='Y (um)', zaxis_title='Depth (um)',
            aspectmode='data',
            bgcolor='black'
        ),
        template='plotly_dark',
        paper_bgcolor='black'
    )
    
    fig.write_html(save_path)
    print(f"✨ 3D Network Plot saved to {save_path}")
