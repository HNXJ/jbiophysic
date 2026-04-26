# src/jbiophysic/models/pipelines/rate_model_pipeline.py
from jbiophysic.common.utils.logging import get_logger

logger = get_logger(__name__)

import jax
import jax.numpy as jnp
import diffrax
from jbiophysic.models.builders.rate_models import EIRateModel
from jbiophysic.common.utils.serialization import safe_save_json

def run_rate_model_simulation():
    logger.info("🚀 Running E/I Rate Model Pipeline: Equinox + Diffrax")
    
    # 1. Initialize model and state
    model = EIRateModel(gain=2.0)
    y0 = jnp.array([0.1, 0.1])
    inputs = jnp.array([1.0, 0.5])
    
    # 2. Setup simulation time
    t0, t1 = 0.0, 200.0
    dt0 = 0.1
    
    # 3. Solver configuration
    logger.info("Configuring Diffrax solver")
    term = diffrax.ODETerm(model)
    solver = diffrax.Tsit5()
    saveat = diffrax.SaveAt(ts=jnp.linspace(t0, t1, 201))
    
    # 4. Execute simulation
    logger.info("Executing JAX-compiled simulation...")
    sol = diffrax.diffeqsolve(
        term, 
        solver, 
        t0, 
        t1, 
        dt0, 
        y0, 
        args=inputs, 
        saveat=saveat
    )
    
    # 5. Result processing
    results = {
        "times": sol.ts.tolist(),
        "v_exc": sol.ys[:, 0].tolist(),
        "v_inh": sol.ys[:, 1].tolist(),
        "metadata": {"model": "EIRateModel", "t_max": t1}
    }
    
    output_path = "pipeline/rate_model_results.json"
    safe_save_json(results, output_path)
    
    logger.info(f"✅ Simulation complete. Steady state E: {sol.ys[-1, 0]:.4f}")
    return sol

if __name__ == "__main__":
    run_rate_model_simulation()
