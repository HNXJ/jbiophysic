#!/usr/bin/env python3
"""
v0.3.6: 100-Neuron Excitatory-Inhibitory Population Tutorial

Purpose
-------
Demonstrate jaxfne v0.3.5+ chainable Configuration API for building and simulating
a 100-neuron mixed E/I (excitatory/inhibitory) population with:
  - 75 excitatory (E) neurons
  - 25 inhibitory (I) neurons
  - Random synaptic connectivity with E→E, E→I, I→E, I→I projections
  - Izhikevich dynamics with "cortical_eig" preset
  - Multimodal readouts: spikes, voltage, source, LFP-proxy, CSD-proxy
  - JSON-safe output bundles with validation metadata

Scientific Status
-----------------
Exploratory computational scaffold. All outputs are proxy readouts for simulation,
validation, and learning workflows. This is not a biological validation or
biophysical calibration. Izhikevich dynamics use native/model current; no
empirical amplitude calibration is claimed.

Usage
-----
From jaxfne repository root:
    PYTHONPATH=. python -m jaxfne.examples.v036_100neuron_ei_population \\
        --out outputs/v036_100neuron_ei_population --duration-ms 500.0

Or from installed jaxfne:
    python -m jaxfne.examples.v036_100neuron_ei_population \\
        --out outputs/v036_100neuron_ei_population
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

try:
    import jaxfne as jtfne
    HAS_JAXFNE = True
except ImportError:
    HAS_JAXFNE = False


def build_ei_configuration(
    n_e: int = 75,
    n_i: int = 25,
    duration_ms: float = 500.0,
    dt_ms: float = 0.1,
    seed: int = 42,
    e_synapse_strength: float = 1.0,
    i_synapse_strength: float = 1.0,
) -> jtfne.Configuration:
    """
    Build a configured E/I population using jaxfne chainable grammar.

    Parameters
    ----------
    n_e : int
        Number of excitatory neurons (default 75)
    n_i : int
        Number of inhibitory neurons (default 25)
    duration_ms : float
        Simulation duration in milliseconds
    dt_ms : float
        Integration timestep in milliseconds
    seed : int
        PRNG seed for deterministic initialization
    e_synapse_strength : float
        Excitatory synaptic weight scale
    i_synapse_strength : float
        Inhibitory synaptic weight scale

    Returns
    -------
    jtfne.Configuration
        Fully configured model ready for construct() and simulate()
    """

    # Start with runtime configuration
    cfg = jtfne.Configuration()
    cfg = cfg.runtime(
        seed=seed,
        dtype="float32",
        duration_ms=duration_ms,
        dt_ms=dt_ms
    )

    # Define cortical column with two cell types: E and I
    # Using "e_i_population" as the column descriptor
    cfg = cfg.column(
        "e_i_population",
        layers=["L2/3"],  # Single layer for simplicity; extend to multi-layer if needed
        n=n_e + n_i  # Total neuron count
    )

    # Define cell type composition
    # Ratio: 75 E, 25 I (standard 3:1 E:I ratio)
    cfg = cfg.cell_types({
        "E": n_e / (n_e + n_i),  # Fraction of excitatory
        "I": n_i / (n_e + n_i),  # Fraction of inhibitory
    })

    # Define connectivity (all-to-all by default; can specify specific patterns)
    # jaxfne handles E→E, E→I, I→E, I→I automatically
    cfg = cfg.connectivity()

    # Set Izhikevich emitter with cortical preset
    cfg = cfg.set_emitter("izhikevich", "cortical_eig")

    # Configure readout probes
    cfg = cfg.probes([
        "MUA-proxy",      # Multi-unit activity proxy (spike count)
        "source-proxy",   # Source current density
        "LFP-proxy",      # Local field potential proxy
        "CSD-proxy",      # Current source density proxy (spatial derivative)
    ])

    return cfg


def main(
    out_dir: str | None = None,
    duration_ms: float = 500.0,
    dt_ms: float = 0.1,
    n_e: int = 75,
    n_i: int = 25,
    seed: int = 42,
) -> None:
    """
    Run 100-neuron E/I population simulation and save outputs.

    Parameters
    ----------
    out_dir : str, optional
        Output directory. If None, uses 'outputs/v036_100neuron_ei_population'
    duration_ms : float
        Simulation duration in milliseconds (default 500)
    dt_ms : float
        Integration timestep (default 0.1 ms)
    n_e : int
        Number of excitatory neurons (default 75)
    n_i : int
        Number of inhibitory neurons (default 25)
    seed : int
        Random seed for reproducibility (default 42)
    """

    if not HAS_JAXFNE:
        raise ImportError(
            "jaxfne is required for this tutorial. "
            "Install with: pip install jaxfne or pip install -e /path/to/jaxfne"
        )

    # Setup output directory
    if out_dir is None:
        out_dir = "outputs/v036_100neuron_ei_population"
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*70}")
    print("jaxfne v0.3.6 Tutorial: 100-Neuron E/I Population")
    print(f"{'='*70}")
    print("Configuration:")
    print(f"  Excitatory neurons (E):   {n_e}")
    print(f"  Inhibitory neurons (I):   {n_i}")
    print(f"  Total neurons:            {n_e + n_i}")
    print(f"  Duration:                 {duration_ms} ms")
    print(f"  Timestep:                 {dt_ms} ms")
    print(f"  Integration steps:        {int(duration_ms / dt_ms)}")
    print(f"  PRNG seed:                {seed}")
    print("\n")

    # Build configuration
    print("Building configuration...")
    cfg = build_ei_configuration(
        n_e=n_e,
        n_i=n_i,
        duration_ms=duration_ms,
        dt_ms=dt_ms,
        seed=seed,
    )
    print("✓ Configuration built")

    # Construct model
    print("Constructing model...")
    model = jtfne.construct(cfg)
    print("✓ Model constructed")

    # Run simulation
    print(f"Running simulation ({duration_ms} ms at {dt_ms} ms/step)...")
    signals = jtfne.simulate(
        model,
        duration_ms=duration_ms,
        dt_ms=dt_ms,
        seed=seed
    )
    print("✓ Simulation complete")

    # Extract and summarize results
    n_timesteps, n_neurons = signals.V_m.shape

    print("\nResults:")
    print(f"  Timesteps:                {n_timesteps}")
    print(f"  Neurons:                  {n_neurons}")
    print(f"  Spike count:              {int(signals.spikes.sum())}")
    print(f"  Mean spike rate:          {signals.spikes.mean() * 1000.0 / dt_ms:.1f} Hz")
    print(f"  Voltage range:            [{signals.V_m.min():.1f}, {signals.V_m.max():.1f}] mV")

    # Save results to JSON-safe output bundle
    output_manifest = {
        "version": "0.3.6",
        "tutorial": "100-neuron E/I population",
        "configuration": {
            "n_excitatory": n_e,
            "n_inhibitory": n_i,
            "n_total": n_e + n_i,
            "duration_ms": duration_ms,
            "dt_ms": dt_ms,
            "n_timesteps": n_timesteps,
            "seed": seed,
        },
        "simulation_results": {
            "spike_count": int(signals.spikes.sum()),
            "mean_spike_rate_hz": float(signals.spikes.mean() * 1000.0 / dt_ms),
            "voltage_min_mv": float(signals.V_m.min()),
            "voltage_max_mv": float(signals.V_m.max()),
            "voltage_mean_mv": float(signals.V_m.mean()),
        },
        "scope_metadata": {
            "truth_mode": "truth_safe_unverified",
            "computational_level": "scaffold",
            "claim_level": "computational_scaffold",
            "physical_amplitude_claim_allowed": False,
            "biological_proof_claim_allowed": False,
            "notes": "All outputs are proxy readouts for simulation and learning workflows. "
                     "No biological validation or empirical calibration claimed."
        },
    }

    # Save manifest
    manifest_path = out_path / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(output_manifest, f, indent=2)
    print(f"\n✓ Manifest saved to {manifest_path}")

    # Save summary statistics
    stats = {
        "spike_raster_shape": list(signals.spikes.shape),
        "voltage_shape": list(signals.V_m.shape),
        "spike_statistics": {
            "total_spikes": int(signals.spikes.sum()),
            "mean_rate_hz": float(signals.spikes.mean() * 1000.0 / dt_ms),
            "per_neuron_rate_hz": [
                float(signals.spikes[:, i].sum() * 1000.0 / (duration_ms))
                for i in range(min(10, n_neurons))  # First 10 neurons as sample
            ]
        },
        "voltage_statistics": {
            "min_mv": float(signals.V_m.min()),
            "max_mv": float(signals.V_m.max()),
            "mean_mv": float(signals.V_m.mean()),
            "std_mv": float(np.std(signals.V_m)),
        }
    }

    stats_path = out_path / "statistics.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"✓ Statistics saved to {stats_path}")

    print(f"\n{'='*70}")
    print(f"Tutorial complete. Outputs saved to: {out_path.resolve()}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="jaxfne v0.3.6 100-Neuron E/I Population Tutorial"
    )
    parser.add_argument(
        "--out",
        type=str,
        default="outputs/v036_100neuron_ei_population",
        help="Output directory (default: outputs/v036_100neuron_ei_population)"
    )
    parser.add_argument(
        "--duration-ms",
        type=float,
        default=500.0,
        help="Simulation duration in milliseconds (default: 500)"
    )
    parser.add_argument(
        "--dt-ms",
        type=float,
        default=0.1,
        help="Integration timestep in milliseconds (default: 0.1)"
    )
    parser.add_argument(
        "--n-e",
        type=int,
        default=75,
        help="Number of excitatory neurons (default: 75)"
    )
    parser.add_argument(
        "--n-i",
        type=int,
        default=25,
        help="Number of inhibitory neurons (default: 25)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="PRNG seed for reproducibility (default: 42)"
    )

    args = parser.parse_args()

    main(
        out_dir=args.out,
        duration_ms=args.duration_ms,
        dt_ms=args.dt_ms,
        n_e=args.n_e,
        n_i=args.n_i,
        seed=args.seed,
    )
