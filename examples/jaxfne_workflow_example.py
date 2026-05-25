"""
Complete example: jaxfne backend workflow in jbiophysic.

This example demonstrates:
1. Model construction with automatic jaxfne support
2. Connectivity analysis with diagnostics
3. Dual-backend simulation
4. Output comparison and analysis
"""

import numpy as np
from dataclasses import replace

from jbiophysic import jtfne


def example_basic_workflow():
    """Basic workflow: construct → diagnose → simulate."""
    print("=" * 70)
    print("EXAMPLE 1: Basic Workflow")
    print("=" * 70)

    # Step 1: Build model with jaxfne auto-included
    print("\n1. Building model with jaxfne support...")
    init = jtfne.JTFNEInitConfig(
        n_neuron_per_column=100,
        seed=42,
        area_order=("V1", "V4", "PFC"),
    )
    model = jtfne.construct(init)
    print(f"   ✓ Model: {len(model.neurons)} neurons")
    print(f"   ✓ Has jaxfne: {hasattr(model, 'eig_network')}")

    # Step 2: Analyze connectivity
    print("\n2. Analyzing connectivity...")
    diag = jtfne.diagnose_connectivity(model.eig_network, model.edges)
    print(f"   Neurons: {diag['n_neurons']}")
    print(f"   Synapses: {diag['n_edges']}")
    print(f"   Connection density: {diag['connection_density']:.2%}")
    print(f"   Excitatory fraction: {diag['excitatory_fraction']:.1%}")
    print(f"   Receptors: {diag['receptor_counts']}")

    # Step 3: Simulate with default config
    print("\n3. Simulating (legacy backend)...")
    cfg = jtfne.default_cfg()
    sim = replace(cfg.sim, n_trials=1, t_ms=500.0, dt_ms=0.5)
    result = jtfne.simulate(model, sim, backend="legacy")
    print(f"   ✓ Trials: {len(result.trials)}")
    print(f"   ✓ V1 spikes shape: {result.trials[0]['V1']['spikes'].shape}")
    print(f"   ✓ V1 LFP shape: {result.trials[0]['V1']['lfp_contacts'].shape}")

    return model, result


def example_backend_comparison():
    """Compare legacy and jaxfne backends."""
    print("\n" + "=" * 70)
    print("EXAMPLE 2: Backend Comparison")
    print("=" * 70)

    # Build model
    init = jtfne.JTFNEInitConfig(n_neuron_per_column=50, seed=123, area_order=("V1",))
    model = jtfne.construct(init)

    cfg = jtfne.default_cfg()
    sim = replace(cfg.sim, n_trials=1, t_ms=200.0, dt_ms=0.5)

    # Simulate with both backends
    print("\n1. Running legacy simulation...")
    result_legacy = jtfne.simulate(model, sim, backend="legacy")
    print(f"   ✓ Backend: legacy")
    print(f"   ✓ Shape: {result_legacy.trials[0]['V1']['spikes'].shape}")

    print("\n2. Running jaxfne simulation...")
    result_jaxfne = jtfne.simulate(model, sim, backend="jaxfne")
    print(f"   ✓ Backend: {result_jaxfne.backend}")
    print(f"   ✓ Shape: {result_jaxfne.trials[0]['V1']['spikes'].shape}")

    # Compare statistics (not exact spike times, which differ by RNG)
    print("\n3. Comparing statistics...")
    legacy_rate = (result_legacy.trials[0]["V1"]["spikes"].sum() /
                   result_legacy.trials[0]["V1"]["spikes"].size)
    jaxfne_rate = (result_jaxfne.trials[0]["V1"]["spikes"].sum() /
                   result_jaxfne.trials[0]["V1"]["spikes"].size)
    print(f"   Legacy firing rate: {legacy_rate:.2%}")
    print(f"   jaxfne firing rate: {jaxfne_rate:.2%}")

    legacy_lfp = result_legacy.trials[0]["V1"]["lfp_contacts"]
    jaxfne_lfp = result_jaxfne.trials[0]["V1"]["lfp_contacts"]
    print(f"   Legacy LFP power (mean): {np.mean(np.abs(legacy_lfp)):.6f}")
    print(f"   jaxfne LFP power (mean): {np.mean(np.abs(jaxfne_lfp)):.6f}")

    return result_legacy, result_jaxfne


def example_receptor_info():
    """Explore receptor kinetics."""
    print("\n" + "=" * 70)
    print("EXAMPLE 3: Receptor Kinetics")
    print("=" * 70)

    receptors = jtfne.get_receptor_info()

    print("\nStandard Receptor Specs:")
    print("-" * 70)
    print(f"{'Receptor':<12} {'Index':<8} {'Sign':<8} {'Tau (ms)':<12} {'E_rev (mV)':<12}")
    print("-" * 70)

    for name, spec in receptors.items():
        print(
            f"{name:<12} {spec['receptor_index']:<8} "
            f"{spec['sign']:+<8} {spec['tau_ms']:<12.1f} {str(spec['reversal_mV']):<12}"
        )

    print("\nReceptor Mapping in jbiophysic:")
    print("-" * 70)
    mapping = {
        "E→E (local exc)": "AMPA",
        "E→PV (feed E→I)": "AMPA",
        "E→SST (feed E→I)": "AMPA",
        "E→VIP (feed E→I)": "AMPA",
        "E→all (feedforward)": "AMPA",
        "E→all (feedback)": "AMPA",
        "PV→all (fast inh)": "GABA_A",
        "SST→all (med inh)": "GABA_A",
        "VIP→all (VIP inh)": "GABA_A",
    }

    for connection, receptor in mapping.items():
        spec = receptors[receptor]
        print(
            f"{connection:<25} → {receptor:<10} "
            f"(tau={spec['tau_ms']:.0f}ms, sign={spec['sign']:+d})"
        )


def example_network_scaling():
    """Explore scaling with network size."""
    print("\n" + "=" * 70)
    print("EXAMPLE 4: Network Scaling")
    print("=" * 70)

    sizes = [20, 50, 100, 200]

    print("\nNetwork statistics by size:")
    print("-" * 70)
    print(f"{'Neurons':<12} {'Edges':<12} {'Density':<12} {'E_frac':<12}")
    print("-" * 70)

    for n_per_col in sizes:
        init = jtfne.JTFNEInitConfig(
            n_neuron_per_column=n_per_col,
            seed=42,
            area_order=("V1", "V4"),
        )
        model = jtfne.construct(init, include_jaxfne=True)
        diag = jtfne.diagnose_connectivity(model.eig_network, model.edges)

        print(
            f"{diag['n_neurons']:<12} {diag['n_edges']:<12} "
            f"{diag['connection_density']:<12.2%} {diag['excitatory_fraction']:<12.1%}"
        )


def example_full_pipeline():
    """Complete end-to-end pipeline with jaxfne."""
    print("\n" + "=" * 70)
    print("EXAMPLE 5: Full Pipeline (Build→Analyze→Simulate→Readout)")
    print("=" * 70)

    # Build
    print("\n[1/5] Constructing network...")
    init = jtfne.JTFNEInitConfig(
        n_neuron_per_column=80,
        seed=999,
        area_order=("V1", "V4"),
    )
    model = jtfne.construct(init)
    print(f"      ✓ {len(model.neurons)} neurons across {len(init.area_order)} areas")

    # Analyze
    print("\n[2/5] Analyzing connectivity...")
    diag = jtfne.diagnose_connectivity(model.eig_network, model.edges)
    print(f"      ✓ {diag['n_edges']} synapses")
    print(f"      ✓ {diag['connection_density']:.1%} sparsity")

    # Simulate
    print("\n[3/5] Running jaxfne simulation...")
    cfg = jtfne.default_cfg()
    sim = replace(cfg.sim, n_trials=2, t_ms=300.0, dt_ms=0.5)
    result = jtfne.simulate(model, sim, backend="jaxfne")
    print(f"      ✓ {len(result.trials)} trials completed")

    # Readout
    print("\n[4/5] Extracting readouts...")
    for trial_idx, trial in enumerate(result.trials):
        print(f"\n      Trial {trial_idx + 1}:")
        for area in init.area_order:
            area_data = trial[area]
            n_neurons = len(area_data["neurons"])
            firing_rate = (
                area_data["spikes"].sum() / area_data["spikes"].size
            )
            print(
                f"        {area}: {n_neurons} neurons, {firing_rate:.1%} FR, "
                f"LFP shape {area_data['lfp_contacts'].shape}"
            )

    # Analyze
    print("\n[5/5] Computing band-pass filters...")
    from jbiophysic.jtfne import _band_profile

    trial = result.trials[0]
    v1_lfp = trial["V1"]["lfp_contacts"]
    dt_ms = trial["dt_ms"]

    freqs = np.linspace(1, 120, 100)
    alpha_band = _band_profile(v1_lfp.T, dt_ms, freqs, (10, 15))
    gamma_band = _band_profile(v1_lfp.T, dt_ms, freqs, (70, 100))

    print(f"      ✓ Alpha (10-15 Hz) power: {alpha_band.mean():.6f}")
    print(f"      ✓ Gamma (70-100 Hz) power: {gamma_band.mean():.6f}")

    return model, result


if __name__ == "__main__":
    print("\n" + "█" * 70)
    print("█" + " " * 68 + "█")
    print("█" + "  jbiophysic + jaxfne: Complete Workflow Examples".center(68) + "█")
    print("█" + " " * 68 + "█")
    print("█" * 70)

    # Run examples
    model1, result1 = example_basic_workflow()
    result_legacy, result_jaxfne = example_backend_comparison()
    example_receptor_info()
    example_network_scaling()
    model5, result5 = example_full_pipeline()

    print("\n" + "█" * 70)
    print("█" + "  ✓ All examples completed successfully!".center(68) + "█")
    print("█" * 70 + "\n")
