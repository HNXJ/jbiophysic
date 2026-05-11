#!/usr/bin/env python3
"""
Plotting utilities for TFNE-Izhikevich laminar E/I simulation outputs.

Generates publication-quality figures from the simulation outputs:
- laminar spike raster (sample neurons)
- population firing rate by layer and cell type
- TFNE extracellular potential snapshots by depth
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_outputs(output_dir: Path) -> dict:
    """Load simulation outputs."""
    outputs = {}
    
    # Load spike data
    spikes_data = np.load(output_dir / "spikes_and_voltage.npz")
    outputs["spikes"] = spikes_data["spikes"]
    outputs["voltage_mV"] = spikes_data["voltage_mV"]
    
    # Load neuron table
    neurons = []
    with open(output_dir / "neuron_table.csv") as f:
        reader = csv.DictReader(f)
        for row in reader:
            neurons.append(row)
    outputs["neurons"] = neurons
    
    # Load population rates
    rates = []
    with open(output_dir / "population_rates_1ms.csv") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rates.append(row)
    outputs["rates"] = rates
    
    # Load TFNE field
    field_data = np.load(output_dir / "tfne_field_snapshots.npz")
    outputs["phi_V"] = field_data["phi_V"]
    
    # Load grid
    grid_data = np.load(output_dir / "tfne_grid.npz")
    outputs["z_m"] = grid_data["z_m"]
    
    return outputs


def plot_laminar_raster(outputs: dict, output_dir: Path, sample_neurons: int = 50) -> None:
    """Plot spike raster for sample neurons across layers."""
    spikes = outputs["spikes"]
    neurons = outputs["neurons"]
    
    # Sample neurons: pick from each layer/type
    layer_map = {n["neuron_id"]: n["layer"] for n in neurons}
    cell_type_map = {n["neuron_id"]: n["cell_type"] for n in neurons}
    
    layer_order = {"superficial": 0, "mid": 1, "deep": 2}
    cell_order = {"E": 0, "PV": 1, "SST": 2}
    
    # Sort by layer then cell type
    neuron_ids = sorted(
        [int(n["neuron_id"]) for n in neurons],
        key=lambda nid: (layer_order.get(layer_map[str(nid)], 999), 
                        cell_order.get(cell_type_map[str(nid)], 999))
    )
    
    # Sample neurons evenly
    sample_ids = neuron_ids[::max(1, len(neuron_ids) // sample_neurons)][:sample_neurons]
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    time_ms = np.arange(spikes.shape[0]) * 0.1
    
    for i, nid in enumerate(sample_ids):
        spike_times = time_ms[spikes[:, nid]]
        ax.scatter(spike_times, [i] * len(spike_times), s=5, alpha=0.6)
    
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel(f"Neuron (sample of {len(sample_ids)})")
    ax.set_title("Laminar Spike Raster (TFNE-Izhikevich E/I 100)")
    ax.grid(True, alpha=0.3)
    
    fig.tight_layout()
    fig.savefig(output_dir / "figures" / "raster.png", dpi=150)
    print("✓ Saved: raster.png")
    plt.close(fig)


def plot_population_rates(outputs: dict, output_dir: Path) -> None:
    """Plot population firing rates by layer and cell type."""
    rates = outputs["rates"]
    
    # Organize by layer and cell type
    layer_rates = {}
    for row in rates:
        layer = row["layer"]
        cell_type = row["cell_type"]
        t0 = float(row["t0_ms"])
        rate_hz = float(row["rate_hz"])
        
        key = (layer, cell_type)
        if key not in layer_rates:
            layer_rates[key] = {"time": [], "rate": []}
        layer_rates[key]["time"].append(t0)
        layer_rates[key]["rate"].append(rate_hz)
    
    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
    
    layers = ["superficial", "mid", "deep"]
    colors = {"E": "C0", "PV": "C1", "SST": "C2"}

    for ax, layer in zip(axes, layers, strict=False):
        for cell_type in ["E", "PV", "SST"]:
            key = (layer, cell_type)
            if key in layer_rates:
                times = layer_rates[key]["time"]
                rates_hz = layer_rates[key]["rate"]
                ax.plot(times, rates_hz, label=f"{cell_type}", color=colors[cell_type], linewidth=2)
        
        ax.set_ylabel(f"{layer.capitalize()} (Hz)")
        ax.legend(loc="upper right")
        ax.grid(True, alpha=0.3)
    
    axes[-1].set_xlabel("Time (ms)")
    fig.suptitle("Population Firing Rate by Layer and Cell Type")
    fig.tight_layout()
    fig.savefig(output_dir / "figures" / "population_rates.png", dpi=150)
    print("✓ Saved: population_rates.png")
    plt.close(fig)


def plot_field_snapshots(outputs: dict, output_dir: Path) -> None:
    """Plot TFNE extracellular potential snapshots by depth."""
    phi_V = outputs["phi_V"]
    z_m = outputs["z_m"]
    
    # Select a few snapshots
    n_snapshots = phi_V.shape[0]
    snap_indices = [0, n_snapshots // 4, n_snapshots // 2, 3 * n_snapshots // 4, n_snapshots - 1]
    
    fig, axes = plt.subplots(1, len(snap_indices), figsize=(15, 3))
    
    for ax_idx, snap_idx in enumerate(snap_indices):
        phi_snap = phi_V[snap_idx]
        
        # Collapse to 2D by taking mean over x
        phi_2d = np.mean(phi_snap, axis=0)
        
        im = axes[ax_idx].imshow(
            phi_2d, aspect="auto", origin="lower",
            extent=[0, z_m[-1] * 1e3, -0.1, 0.1],
            cmap="RdBu_r"
        )
        axes[ax_idx].set_title(f"t={snap_idx*10:.0f} ms")
        axes[ax_idx].set_xlabel("Depth (mm)")
        if ax_idx == 0:
            axes[ax_idx].set_ylabel("Radial (mm)")
        plt.colorbar(im, ax=axes[ax_idx], label="V")
    
    fig.suptitle("TFNE Extracellular Potential φ (V) Over Time")
    fig.tight_layout()
    fig.savefig(output_dir / "figures" / "field_snapshots.png", dpi=150)
    print("✓ Saved: field_snapshots.png")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("outputs/tfne_izhikevich_laminar_ei100"))
    parser.add_argument("--sample-neurons", type=int, default=50)
    args = parser.parse_args()
    
    out = args.out
    if not out.exists():
        raise FileNotFoundError(f"Output directory not found: {out}")
    
    # Create figures directory
    fig_dir = out / "figures"
    fig_dir.mkdir(exist_ok=True)
    
    print(f"Loading outputs from: {out}")
    outputs = load_outputs(out)
    
    print("Generating figures...")
    plot_laminar_raster(outputs, out, sample_neurons=args.sample_neurons)
    plot_population_rates(outputs, out)
    plot_field_snapshots(outputs, out)
    
    print(f"\nFigures saved to: {fig_dir.resolve()}")


if __name__ == "__main__":
    main()
