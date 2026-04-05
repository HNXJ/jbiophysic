# pipeline/generate_phase_plot.py
import matplotlib.pyplot as plt
import numpy as np
import json

def plot_publication_phase_diagram(data_path="output/phase_data.json"):
    """
    Axis 13 Visual Visualization:
    Generating the PV/SST oscillatory phase landscape (Figure 5).
    """
    with open(data_path, "r") as f:
        data = json.load(f)
    
    a_pv = np.array(data["a_pv"])
    a_sst = np.array(data["a_sst"])
    
    metrics = ["gamma", "beta", "omission_beta"]
    titles = ["Panel A: Gamma Power (FF/Precision)", 
              "Panel B: Beta Power (FB/Feedback)",
              "Panel C: Omission-Beta Increase"]
    
    plt.style.use('dark_background')
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    for i, (metric, title) in enumerate(zip(metrics, titles)):
        im = axes[i].imshow(data[metric], extent=[0.0, 2.0, 0.0, 2.0], 
                            origin='lower', cmap='magma', aspect='auto')
        
        axes[i].set_xlabel(r"PV Scaling ($\alpha_{PV}$)", fontsize=10, color='#CFB87C')
        axes[i].set_ylabel(r"SST Scaling ($\alpha_{SST}$)", fontsize=10, color='#CFB87C')
        axes[i].set_title(title, fontsize=12, fontweight='bold', color='#CFB87C')
        
        cbar = fig.colorbar(im, ax=axes[i])
        cbar.ax.tick_params(labelsize=8)
        
        # Highlight the "Balanced Regime" in Panel C
        if metric == "omission_beta":
            axes[i].scatter([1.0], [1.0], color='#00FF00', s=100, marker='*', label="Balanced")
            axes[i].legend(fontsize=8)
    
    plt.tight_layout()
    plt.savefig("output/phase_diagram_3panel.png", dpi=300, bbox_inches='tight')
    print("✅ Figure 5 (3-Panel Phase Diagram) generated at output/phase_diagram_3panel.png")

if __name__ == "__main__":
    plot_publication_phase_diagram()
