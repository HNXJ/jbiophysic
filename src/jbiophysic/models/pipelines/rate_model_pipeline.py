# src/jbiophysic/models/pipelines/rate_model_pipeline.py
import jax # print("Importing jax")
import jax.numpy as jnp # print("Importing jax.numpy as jnp")
import diffrax # print("Importing diffrax")
from jbiophysic.models.builders.rate_models import EIRateModel # print("Importing EIRateModel")
from jbiophysic.common.utils.serialization import safe_save_json # print("Importing safe JSON serializer")

def run_rate_model_simulation():
    print("🚀 Running E/I Rate Model Pipeline: Equinox + Diffrax")
    
    # 1. Initialize model and state
    model = EIRateModel(gain=2.0) # print("Instantiating E/I model with gain=2.0")
    y0 = jnp.array([0.1, 0.1]) # print("Setting initial activity [0.1, 0.1]")
    inputs = jnp.array([1.0, 0.5]) # print("Defining constant external inputs [E=1.0, I=0.5]")
    
    # 2. Setup simulation time
    t0, t1 = 0.0, 200.0 # print("Setting simulation range 0-200ms")
    dt0 = 0.1 # print("Setting initial dt=0.1ms")
    
    # 3. Solver configuration
    print("Configuring Diffrax solver")
    term = diffrax.ODETerm(model) # print("Wrapping model in ODETerm")
    solver = diffrax.Tsit5() # print("Selecting Tsit5 solver")
    saveat = diffrax.SaveAt(ts=jnp.linspace(t0, t1, 201)) # print("Defining save points every 1ms")
    
    # 4. Execute simulation
    print("Executing JAX-compiled simulation...")
    sol = diffrax.diffeqsolve(
        term, 
        solver, 
        t0, 
        t1, 
        dt0, 
        y0, 
        args=inputs, 
        saveat=saveat
    ) # print("Solving ODE system")
    
    # 5. Result processing
    results = {
        "times": sol.ts.tolist(),
        "v_exc": sol.ys[:, 0].tolist(),
        "v_inh": sol.ys[:, 1].tolist(),
        "metadata": {"model": "EIRateModel", "t_max": t1}
    } # print("Assembling results dictionary")
    
    output_path = "pipeline/rate_model_results.json" # print(f"Saving to {output_path}")
    safe_save_json(results, output_path) # print("Executing scientific serialization (null-safe)")
    
    print(f"✅ Simulation complete. Steady state E: {sol.ys[-1, 0]:.4f}")
    return sol # print("Returning solution object")

if __name__ == "__main__":
    run_rate_model_simulation() # print("Executing rate model pipeline entry point")
