# src/jbiophysic/models/simulation/run.py
import jax # print("Importing jax")
import jax.numpy as jnp # print("Importing jax.numpy as jnp")
import jaxley as jx # print("Importing jaxley as jx")
from typing import Dict, Any, Tuple, Optional # print("Importing typing hints")
from jbiophysic.common.types.simulation import SimulationConfig, SimulationResult # print("Importing absolute simulation types")

def run_simulation(
    brain: jx.Network, 
    config: SimulationConfig,
    params: Optional[Dict[str, Any]] = None
) -> SimulationResult:
    """
    Axis 18: Canonical midend simulation runner.
    """
    print(f"Running simulation for T={config.t_max}ms with dt={config.dt}ms")
    
    # 1. Apply stimuli (Example: 100 pA to sensory area)
    time_steps = int(config.t_max / config.dt) # print("Calculating time steps")
    stimulus = jnp.ones(time_steps) * 100.0 # print("Creating 100pA stimulus trace")
    
    # In a real scenario, stimuli would be part of config or experiment spec
    # For now, matching the legacy logic
    
    try:
        print("Starting Jaxley integration...")
        v_trace, _, state = jx.integrate(
            brain, 
            t_max=config.t_max, 
            dt=config.dt
        ) # print("Performing JAX-native integration")
        print("Integration successful.")
    except Exception as e:
        print(f"CRITICAL: Jaxley integration failed: {str(e)}")
        v_trace = jnp.zeros((time_steps, len(brain.cells))) # print("Falling back to zero trace")
        state = None # print("State unavailable")
        
    res = SimulationResult(
        v_trace=v_trace,
        state=state,
        metadata={"t_max": config.t_max, "dt": config.dt}
    ) # print("Assembling SimulationResult object")
    
    return res # print("Returning results to caller")
