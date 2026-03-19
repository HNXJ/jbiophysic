import pandas as pd
import numpy as np
import plotly.graph_objects as go
import os

def plot_ipnoise_sweep():
    csv_path = '/Users/hamednejat/workspace/Computational/jbiophysics/systems/actions/ipnoise_sweep_results.csv'
    save_dir = '/Users/hamednejat/workspace/Analysis/nwb/oxm/figures/noise_sweep'
    os.makedirs(save_dir, exist_ok=True)

    if not os.path.exists(csv_path):
        print(f"CSV not found: {csv_path}")
        return

    df = pd.read_csv(csv_path)
    
    # Pivot for Heatmap
    # X = poisson_l, Y = pulse_amp
    pivot_afr = df.pivot(index='pulse_amp', columns='poisson_l', values='mean_afr')
    pivot_kappa = df.pivot(index='pulse_amp', columns='poisson_l', values='kappa')

    # --- Plotly Interactive ---
    # Heatmap 1: AFR
    fig_afr = go.Figure(data=go.Heatmap(
        z=pivot_afr.values,
        x=pivot_afr.columns,
        y=pivot_afr.index,
        colorscale='Viridis',
        colorbar=dict(title='Mean AFR (Hz)')
    ))
    fig_afr.update_layout(title='IPnoise Sweep: Mean AFR', xaxis_title='Poisson Lambda (ms)', yaxis_title='Pulse Amplitude (nA)')
    fig_afr.write_html(os.path.join(save_dir, 'ipnoise_afr_sweep.html'))

    # Heatmap 2: Kappa
    fig_kappa = go.Figure(data=go.Heatmap(
        z=pivot_kappa.values,
        x=pivot_kappa.columns,
        y=pivot_kappa.index,
        colorscale='Magma',
        colorbar=dict(title='Kappa')
    ))
    fig_kappa.update_layout(title='IPnoise Sweep: Population Kappa', xaxis_title='Poisson Lambda (ms)', yaxis_title='Pulse Amplitude (nA)')
    fig_kappa.write_html(os.path.join(save_dir, 'ipnoise_kappa_sweep.html'))

    # --- Matplotlib SVG ---
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(1, 2, figsize=(15, 6), facecolor='white')
    
    im1 = axs[0].imshow(pivot_afr.values, origin='lower', aspect='auto', cmap='viridis')
    axs[0].set_title('Mean AFR (Hz)')
    plt.colorbar(im1, ax=axs[0])
    
    im2 = axs[1].imshow(pivot_kappa.values, origin='lower', aspect='auto', cmap='magma')
    axs[1].set_title('Population Kappa')
    plt.colorbar(im2, ax=axs[1])

    for ax in axs:
        ax.set_xticks(np.arange(len(pivot_afr.columns)))
        ax.set_xticklabels(pivot_afr.columns)
        ax.set_yticks(np.arange(len(pivot_afr.index)))
        ax.set_yticklabels(pivot_afr.index)
        ax.set_xlabel('Poisson Lambda (ms)')
        ax.set_ylabel('Pulse Amplitude (nA)')

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'ipnoise_sweep_summary.svg'))
    print(f"✨ Figures saved to {save_dir}")

if __name__ == "__main__":
    plot_ipnoise_sweep()
