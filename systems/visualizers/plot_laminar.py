import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import re
import os

def generate_laminar_plots():
    log_path = '/Users/hamednejat/workspace/Computational/jbiophysics/systems/actions/laminar_100_opt_log.txt'
    save_dir = '/Users/hamednejat/workspace/media/figures/optimizer_tests'
    os.makedirs(save_dir, exist_ok=True)

    if not os.path.exists(log_path):
        print(f"Log not found: {log_path}")
        return

    with open(log_path, 'r') as f:
        log_content = f.read()

    # Regex to extract metrics
    # Format: Trial {i} | Loss: {loss_val:.4f} | Alpha: {opt_state.a:.4f} | Kappa: {kappa:.4f} | AFR: {afr:.2f}
    pattern = r"Trial (\d+) \| Loss: ([\d\.]+) \| Alpha: ([\d\.]+) \| Kappa: ([\-\d\.]+) \| AFR: ([\d\.]+)"
    matches = re.findall(pattern, log_content)
    
    if not matches:
        print("No valid metrics found in log.")
        return

    trials = [int(m[0]) for m in matches]
    loss = [float(m[1]) for m in matches]
    alpha = [float(m[2]) for m in matches]
    kappa = [float(m[3]) for m in matches]
    afr = [float(m[4]) for m in matches]

    # --- Plotly Dashboard ---
    fig = make_subplots(rows=2, cols=2, 
                        subplot_titles=("Loss Trajectory", "Adaptive Alpha", "Avg Firing Rate (Target: 5-10Hz)", "Kappa Synchrony (Target: 0)"))

    fig.add_trace(go.Scatter(x=trials, y=loss, name="Loss", mode='lines+markers', line=dict(color='gold')), row=1, col=1)
    fig.add_trace(go.Scatter(x=trials, y=alpha, name="Alpha", mode='lines', line=dict(color='cyan')), row=1, col=2)
    fig.add_trace(go.Scatter(x=trials, y=afr, name="AFR", mode='lines+markers', line=dict(color='white')), row=2, col=1)
    fig.add_trace(go.Scatter(x=trials, y=kappa, name="Kappa", mode='lines', line=dict(color='red')), row=2, col=2)

    # Adding target ranges for clarity
    fig.add_hline(y=10.0, line_dash="dash", line_color="green", row=2, col=1)
    fig.add_hline(y=5.0, line_dash="dash", line_color="green", row=2, col=1)
    fig.add_hline(y=0.0, line_dash="dash", line_color="green", row=2, col=2)

    fig.update_layout(height=900, width=1200, title_text="100-Neuron Laminar Column Optimization (AGSDR v2)", 
                      template="plotly_dark", paper_bgcolor='black', plot_bgcolor='black')
    
    html_path = os.path.join(save_dir, "laminar_100_interactive.html")
    fig.write_html(html_path)
    
    # Save static PNG for quick preview
    import matplotlib.pyplot as plt
    plt.style.use('dark_background')
    fig_static, axs = plt.subplots(2, 2, figsize=(15, 10))
    axs[0, 0].plot(trials, loss, color='#CFB87C'); axs[0, 0].set_title('Loss')
    axs[0, 1].plot(trials, alpha, color='cyan'); axs[0, 1].set_title('Adaptive Alpha')
    axs[1, 0].plot(trials, afr, color='white'); axs[1, 0].set_title('AFR (Hz)')
    axs[1, 0].axhline(10, ls='--', color='green'); axs[1, 0].axhline(5, ls='--', color='green')
    axs[1, 1].plot(trials, kappa, color='red'); axs[1, 1].set_title('Kappa')
    axs[1, 1].axhline(0, ls='--', color='green')
    
    png_path = os.path.join(save_dir, "laminar_100_summary.png")
    plt.tight_layout()
    plt.savefig(png_path, dpi=300)
    
    print(f"✨ SUCCESS: Dashboard saved to {html_path}")
    print(f"🖼️ SUCCESS: Static summary saved to {png_path}")

if __name__ == "__main__":
    generate_laminar_plots()
