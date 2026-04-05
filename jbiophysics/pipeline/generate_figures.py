# pipeline/generate_figures.py
import matplotlib.pyplot as plt
import numpy as np
import json

def plot_publication_phase_diagram(data_path="output/phase_data.json"):
    """
    Axis 13 Visual Visualization:
    Generating the 4-panel Nature-style layout for the inhibitory phase landscape.
    """
    with open(data_path, "r") as f:
        data = json.load(f)
    
    a_pv = np.array(data["a_pv"])
    a_sst = np.array(data["a_sst"])
    
    metrics = ["gamma", "beta", "omission_beta", "error"]
    titles = ["Panel A: Gamma (FF-Precision)", 
              "Panel B: Beta (FB-Feedback)",
              "Panel C: Omission-Beta Effect",
              "Panel D: PC Error Signals"]
    
    plt.style.use('dark_background')
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes = axes.flatten()
    
    for i, (metric, title) in enumerate(zip(metrics, titles)):
        im = axes[i].imshow(data[metric], extent=[0.0, 2.0, 0.0, 2.0], 
                            origin='lower', cmap='magma', aspect='auto')
        
        axes[i].set_xlabel(r"PV Scaling ($\alpha_{PV}$)", fontsize=11, color='#CFB87C')
        axes[i].set_ylabel(r"SST Scaling ($\alpha_{SST}$)", fontsize=11, color='#CFB87C')
        axes[i].set_title(title, fontsize=13, fontweight='bold', color='#CFB87C')
        
        cbar = fig.colorbar(im, ax=axes[i])
        cbar.ax.tick_params(labelsize=9)
        
        # Mark Balanced Point (Axis 13+)
        if metric == "omission_beta":
            axes[i].scatter([1.0], [1.0], color='#00FF00', s=100, marker='*', label="Balanced")
            axes[i].legend(fontsize=9)
    
    plt.tight_layout()
    plt.savefig("output/figure5_phase_diagram.png", dpi=300, bbox_inches='tight')
    print("✅ Figure 5 (4-Panel Publication Figure) generated at output/figure5_phase_diagram.png")

if __name__ == "__main__":
    plot_publication_phase_diagram()
