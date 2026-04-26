# src/jbiophysic/models/pipelines/tracer_bullet.py
from jbiophysic.common.utils.logging import get_logger

logger = get_logger(__name__)

import jax
import jax.numpy as jnp
import diffrax
from jbiophysic.models.builders.tracer_neuron import TracerLIF

def run_tracer_bullet():
    logger.info("🚀 Running Tracer Bullet: Equinox + Diffrax Integration")
    
    # 1. Instantiate the Equinox model
    neuron = TracerLIF(tau_m=20.0, v_rest=-70.0)
    
    # 2. Define simulation parameters
    t0, t1 = 0.0, 100.0
    dt0 = 0.1
    y0 = -70.0
    i_ext = 25.0
    
    # 3. Setup Diffrax ODE solver
    logger.info("Setting up Diffrax ODE solver...")
    term = diffrax.ODETerm(neuron)
    solver = diffrax.Tsit5()
    saveat = diffrax.SaveAt(ts=jnp.linspace(t0, t1, 101))
    
    # 4. Execute JIT-compiled simulation
    logger.info("Executing simulation (first run triggers JIT)...")
    sol = diffrax.diffeqsolve(
        term, 
        solver, 
        t0, 
        t1, 
        dt0, 
        y0, 
        args=i_ext, 
        saveat=saveat
    )
    
    # 5. Extract results
    times = sol.ts
    voltages = sol.ys
    
    logger.info(f"✅ Tracer Bullet Complete. Final Voltage: {voltages[-1]:.2f}mV")
    return times, voltages

if __name__ == "__main__":
    run_tracer_bullet()
