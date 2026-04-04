import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Any

# Aesthetic Constants (Axis 8: Golden Dark)
BG = "#0D0D0F"
GOLD = "#CFB87C"
WHITE = "#E8E8E8"
CYAN = "#00FFFF"
VIOLET = "#9400D3"
TEAL = "#00FFCC"
ORANGE = "#FF8C00"

def setup_madelane_style():
    """Sets the Madelane Golden Dark style (Axis 8)."""
    plt.style.use('dark_background')
    plt.rcParams.update({
        'axes.facecolor': BG,
        'figure.facecolor': BG,
        'axes.edgecolor': GOLD,
        'grid.color': '#2D2D2D',
        'text.color': GOLD,
        'axes.labelcolor': GOLD,
        'xtick.color': GOLD,
        'ytick.color': GOLD,
        'font.family': 'sans-serif'
    })

def plot_panel_a_tfr(t, f, Sxx_db, onset_ms=0, title="Time-Frequency Response (TFR)"):
    """Axis 8: Panel A - TFR matches poster panels exactly."""
    setup_madelane_style()
    fig, ax = plt.subplots(figsize=(10, 4))
    
    # Gouraud shading for smoothness
    pc = ax.pcolormesh(t, f, Sxx_db, shading='gouraud', cmap='magma')
    ax.set_ylim(0, 100)
    
    # Omission onset line (Axis 6)
    ax.axvline(onset_ms, linestyle='--', color='white', alpha=0.5)
    
    ax.set_title(title, fontsize=14, pad=10)
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Frequency (Hz)")
    
    cbar = plt.colorbar(pc, ax=ax)
    cbar.set_label("Power (dB)", color=GOLD)
    return fig

def plot_panel_b_corr_matrix(area_signals: List[np.ndarray], area_names: List[str], title="Spectral Correlation"):
    """Axis 8: Panel B - Area-by-area correlation matrix."""
    setup_madelane_style()
    data = np.array(area_signals)
    corr = np.corrcoef(data)
    
    fig, ax = plt.subplots(figsize=(8, 8))
    im = ax.imshow(corr, cmap='viridis', vmin=-1, vmax=1)
    
    ax.set_xticks(range(len(area_names)))
    ax.set_xticklabels(area_names, rotation=45)
    ax.set_yticks(range(len(area_names)))
    ax.set_yticklabels(area_names)
    
    ax.set_title(title, fontsize=14, pad=10)
    plt.colorbar(im, ax=ax, label="Correlation (R)")
    return fig

def plot_panel_c_band_comparison(results: Dict[str, Dict[str, float]], bands: List[str] = ["theta", "alpha", "beta", "gamma"]):
    """Axis 8: Panel C - Band power across hierarchical levels."""
    setup_madelane_style()
    fig, ax = plt.subplots(figsize=(10, 4))
    
    colors = {"theta": CYAN, "alpha": TEAL, "beta": VIOLET, "gamma": GOLD}
    
    # Results expected: {area_name: {band: power}}
    areas = list(results.keys())
    for b in bands:
        powers = [results[area].get(b, 0.0) for area in areas]
        ax.plot(areas, powers, marker='o', label=b, color=colors.get(b, ORANGE))
        
    ax.set_title("Band Power Evolution Across Hierarchy", fontsize=14, pad=10)
    ax.set_ylabel("Power (a.u.)")
    ax.set_xlabel("Hierarchical Level (Low → High)")
    ax.legend()
    return fig

def export_poster_panel(fig, name: str, output_dir: str = "./output"):
    """Exports production-ready PDF/PNG for figures (Axis 8)."""
    import os
    os.makedirs(output_dir, exist_ok=True)
    fig.savefig(f"{output_dir}/{name}.pdf", transparent=False, dpi=300)
    fig.savefig(f"{output_dir}/{name}.png", transparent=False, dpi=300)
    print(f"🖼️ Poster panel exported → {output_dir}/{name}")
