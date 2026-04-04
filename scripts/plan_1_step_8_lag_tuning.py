import jax
import jax.numpy as jnp
import jaxley as jx
import numpy as np
import os
import sys

# Add current directory to path
sys.path.insert(0, os.getcwd())

from compose import NetBuilder, OptimizerFacade

def run_lag_tuning():
    """
    Step 8: Physiological Lag Tuning.
    Adjust g_AMPA to match the empirical 40-60ms sensory lag.
    """
    # 1. Build the Hierarchy
    builder = (NetBuilder(seed=42)
        .add_population("E", n=20, cell_type="pyr", area="V1")
        .add_population("PV", n=5, cell_type="pv", area="V1")
        .add_population("E", n=20, cell_type="pyr", area="V2")
        .connect("E", "PV", synapse="AMPA", p=0.2, area="V1")
        .connect("PV", "E", synapse="GABAa", p=0.4, area="V1")
        # Inter-areal FF V1 -> V2
        .connect("V1.E", "V2.E", synapse="AMPA", p=0.1, g=0.5)
        .make_trainable(["gAMPA"]))

    net = builder.build()
    
    # 2. Define Stimulus (Onset at 500ms)
    t_max = 1000.0
    dt = 0.025
    times = jnp.arange(0, t_max, dt)
    stim_onset = 500.0
    stim_duration = 50.0
    stim_amp = 0.5 # nA
    
    # Square pulse stimulus
    stim_vector = jnp.where((times >= stim_onset) & (times < stim_onset + stim_duration), stim_amp, 0.0)
    
    # Apply stimulus to V1.E
    v1_e_start, v1_e_end = builder.population_offsets["V1.E"]
    v1_e_indices = list(range(v1_e_start, v1_e_end))
    net.cell(v1_e_indices).branch(0).loc(0.0).stimulate(stim_vector)

    # 3. Configure Optimizer (AGSDR + Adam)
    facade = (OptimizerFacade(net, method="AGSDR", lr=1e-3)
              .set_pop_offsets(builder.population_offsets)
              # Global stability
              .set_constraints(firing_rate=(1.0, 50.0), kappa_max=0.1)
              # Target 50ms Lag in V1.E (peak at 550ms)
              .set_lag_constraint("V1.E", target_ms=50.0, stimulus_onset_ms=stim_onset)
              # V1.E Firing rate target during stimulus
              .set_pop_constraints("V1.E", firing_rate=(5.0, 20.0)))
    
    print("🚀 Starting Step 8: Physiological Lag Tuning...")
    report = facade.run(epochs=50, dt=dt, t_max=t_max)
    
    # 4. Export Results
    report.save_json("jbiophysics/results/data/step_8_lag_tuning.json")
    print(f"✅ Lag Tuning Complete. Final Loss: {report.metadata['history']['loss'][-1]:.4f}")

if __name__ == "__main__":
    run_lag_tuning()
