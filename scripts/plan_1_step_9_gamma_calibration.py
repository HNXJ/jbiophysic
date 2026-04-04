import jax
import jax.numpy as jnp
import jaxley as jx
import numpy as np
import os
import sys

# Add current directory to path
sys.path.insert(0, os.getcwd())

from compose import NetBuilder, OptimizerFacade
from systems.visualizers.compute_psd import compute_psd

def run_gamma_calibration():
    """
    Step 9: PING/ING Rhythm Calibration.
    Optimize local g_PV->E and g_E->PV for a stable 41 Hz Gamma rhythm in V1.
    """
    # 1. Build the V1-only microcircuit
    builder = (NetBuilder(seed=42)
        .add_population("E", n=50, cell_type="pyr", area="V1")
        .add_population("PV", n=15, cell_type="pv", area="V1")
        .add_population("SST", n=10, cell_type="sst", area="V1")
        # Local E/I motifs
        .connect("E", "PV", synapse="AMPA", p=0.2, area="V1")
        .connect("PV", "E", synapse="GABAa", p=0.4, area="V1")
        .connect("E", "E", synapse="AMPA", p=0.1, area="V1")
        .connect("SST", "E", synapse="GABAb", p=0.2, area="V1")
        .make_trainable(["gAMPA", "gGABAa", "gGABAb"]))

    net = builder.build()
    
    # 2. Define Target PSD (Gaussian peak at 41 Hz)
    dt = 0.025
    t_max = 1000.0
    freq_axis = jnp.linspace(1.0, 100.0, 100)
    
    # Gaussian peak centered at 41Hz
    target_psd = jnp.exp(-((freq_axis - 41.0)**2) / (2 * 5.0**2))
    target_psd = target_psd / (jnp.max(target_psd) + 1e-12) # Normalize to 1.0 peak
    
    # 3. Configure Optimizer (AGSDR + Adam)
    facade = (OptimizerFacade(net, method="AGSDR", lr=1e-3)
              .set_pop_offsets(builder.population_offsets)
              # Global stability and asynchronous-irregular regime
              .set_constraints(firing_rate=(5.0, 40.0), kappa_max=0.1)
              # Set spectral target for V1.E (matching Gamma peak)
              .set_target(target_psd))
    
    print("🚀 Starting Step 9: 41 Hz Gamma Rhythm Calibration...")
    report = facade.run(epochs=100, dt=dt, t_max=t_max)
    
    # 4. Export Results
    report.save_json("jbiophysics/results/data/step_9_gamma_calibration.json")
    print(f"✅ Gamma Calibration (41 Hz) Complete. Final Loss: {report.metadata['history']['loss'][-1]:.4f}")

if __name__ == "__main__":
    run_gamma_calibration()
