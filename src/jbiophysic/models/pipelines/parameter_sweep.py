# src/jbiophysic/models/pipelines/parameter_sweep.py
"""
DynaSim-Parity Tutorial: Exploring Firing Rate Modulation via A-type Potassium.

This script demonstrates the 'Worked Example' logic from Sherfey et al. (2018):
1. Building a biophysical neuron with multiple currents (HH + IA).
2. Defining a parameter sweep grid.
3. Running a batch simulation.
4. Extracting the firing rate relationship (f-I curve).
"""

from jbiophysic.common.utils.logging import get_logger
logger = get_logger(__name__)

import jax.numpy as jnp
import jaxley as jx
from jbiophysic.core.mechanisms.channels.hh_base import HH
from jbiophysic.core.mechanisms.channels.extra_currents import IA
from jbiophysic.models.simulation.batch import run_batch_simulation
from jbiophysic.common.types.simulation import SimulationConfig

def run_ia_sweep():
    logger.info("🎬 Starting DynaSim-style Parameter Sweep: IA Conductance")
    
    # 1. Build a single compartment neuron with HH and IA
    soma = jx.Branch(ncomp=1)
    cell = jx.Cell(soma)
    cell.insert(HH())
    cell.insert(IA())
    
    # 2. Define the Sweep Grid
    # We vary gka (A-type conductance) across 5 values
    gka_values = jnp.linspace(0.0, 0.05, 5)
    param_grid = {"gka": gka_values}
    
    # 3. Configure Batch Simulation
    config = SimulationConfig(t_max=200.0, dt=0.025)
    
    # 4. Run Batch
    results = run_batch_simulation(cell, config, param_grid)
    
    # 5. Extract Firing Rates (Simple thresholding)
    threshold = -20.0
    for res in results:
        v = res.v_trace[0] # Single compartment
        spikes = jnp.where((v[:-1] < threshold) & (v[1:] >= threshold))[0]
        firing_rate = len(spikes) / (config.t_max / 1000.0)
        
        gka = res.metadata["gka"]
        logger.info(f"gka={gka:.3f} | Firing Rate: {firing_rate:.2f} Hz")

if __name__ == "__main__":
    run_ia_sweep()
