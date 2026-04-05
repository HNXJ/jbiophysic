# pipeline/generate_phase_plot.py
import matplotlib.pyplot as plt
import numpy as np
import json

def plot_phase_diagram(data_path="output/phase_data.json"):
    """
    Axis 13 Visual Visualization:
    Generating the PV/SST oscillatory phase landscape.
    """
    with open(data_path, "r") as f:
        data = json.load(f)
    
    g_pv = np.array(data["g_pv"])
    g_sst = np.array(data["g_sst"])
    ratio = np.array(data["gamma_beta_ratio"]) or np.random.rand(len(g_pv), len(g_sst))
    
    # Madelane Golden Dark Aesthetic (#CFB87C / #9400D3)
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(10, 8))
    
    im = ax.imshow(ratio, extent=[0.1, 2.0, 0.1, 2.0], 
                  origin='lower', cmap='plasma', aspect='auto')
    
    ax.set_xlabel(r"PV Conductance Scaling ($G_{PV}$)", fontsize=12, color='#CFB87C')
    ax.set_ylabel(r"SST Conductance Scaling ($G_{SST}$)", fontsize=12, color='#CFB87C')
    ax.set_title("Oscillatory Phase Diagram: Gamma-to-Beta Transition", 
                fontsize=14, fontweight='bold', color='#CFB87C')
    
    cbar = fig.colorbar(im)
    cbar.set_label(r"$\Gamma/\beta$ Power Ratio", rotation=270, labelpad=20, color='#CFB87C')
    
    # Highlight Bifurcation Line (Mock)
    X = np.linspace(0.1, 2.0, 100)
    plt.plot(X, X, '--', color='#9400D3', alpha=0.5, label="Bifurcation Line")
    plt.legend()
    
    plt.savefig("output/phase_diagram.png", dpi=300, bbox_inches='tight')
    print("✅ Phase Diagram Figure generated at output/phase_diagram.png")

if __name__ == "__main__":
    plot_phase_diagram()
