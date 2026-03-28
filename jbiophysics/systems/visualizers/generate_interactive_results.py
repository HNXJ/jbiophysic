import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import re
import os

def generate_interactive_comparison():
    log_path = '/Users/hamednejat/workspace/Computational/jbiophysics/systems/actions/baseline_opt_log.txt'
    save_dir = '/Users/hamednejat/workspace/media/figures/optimizer_tests'
    os.makedirs(save_dir, exist_ok=True)

    with open(log_path, 'r') as f:
        log_content = f.read()

    # Regex to extract metrics
    # Format: Trial {i} | Loss: {loss_val:.4f} | Alpha: {state.a:.4f} | Kappa: {kappa:.4f} | AFR: {jnp.mean(afr):.2f}
    pattern = r"Trial (\d+) \| Loss: ([\d\.]+) \| Alpha: ([\d\.]+) \| Kappa: ([\-\d\.]+) \| AFR: ([\d\.]+)"
    
    # Split log by "Training with" to separate GSDR and AGSDR
    sections = log_content.split("🚀 Training with ")
    
    fig = make_subplots(rows=2, cols=2, subplot_titles=("Loss Trajectory", "Alpha (Mixing)", "Avg Firing Rate", "Kappa Synchrony"))

    for section in sections[1:]:
        name = section.split("...")[0].strip()
        matches = re.findall(pattern, section)
        if not matches: continue
        
        trials = [int(m[0]) for m in matches]
        loss = [float(m[1]) for m in matches]
        alpha = [float(m[2]) for m in matches]
        kappa = [float(m[3]) for m in matches]
        afr = [float(m[4]) for m in matches]

        fig.add_trace(go.Scatter(x=trials, y=loss, name=f"{name} Loss", mode='lines'), row=1, col=1)
        fig.add_trace(go.Scatter(x=trials, y=alpha, name=f"{name} Alpha", mode='lines'), row=1, col=2)
        fig.add_trace(go.Scatter(x=trials, y=afr, name=f"{name} AFR", mode='lines'), row=2, col=1)
        fig.add_trace(go.Scatter(x=trials, y=kappa, name=f"{name} Kappa", mode='lines'), row=2, col=2)

    fig.update_layout(height=900, width=1200, title_text="Optimizer Benchmarking: GSDR vs AGSDR v2", template="plotly_white")
    
    save_path = os.path.join(save_dir, "baseline_10_ei_interactive.html")
    fig.write_html(save_path)
    print(f"✨ SUCCESS: Interactive dashboard saved to {save_path}")

if __name__ == "__main__":
    generate_interactive_comparison()
