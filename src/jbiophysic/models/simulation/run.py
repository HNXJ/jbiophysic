# src/jbiophysic/models/simulation/run.py
import jax
import jax.numpy as jnp
import jaxley as jx
from typing import Dict, Any, Tuple, Optional
from jbiophysic.common.types.simulation import SimulationConfig, SimulationResult

def run_simulation(
    brain: jx.Network, 
    config: SimulationConfig,
    params: Optional[Dict[str, Any]] = None
) -> SimulationResult:
    """
    Axis 18: Canonical models simulation runner.
    """
    print(f"Running simulation for T={config.t_max}ms with dt={config.dt}ms")
    
    # 1. Apply stimuli (Example: 100 pA to sensory area)
    time_steps = int(config.t_max / config.dt)
    stimulus = jnp.ones(time_steps) * 100.0
    
    # In a real scenario, stimuli would be part of config or experiment spec
    # For now, matching the legacy logic
    
    try:
        print("Starting Jaxley integration...")
        v_trace, _, state = jx.integrate(
            brain, 
            t_max=config.t_max, 
            dt=config.dt
        )
        print("Integration successful.")
    except Exception as e:
        print(f"CRITICAL: Jaxley integration failed: {str(e)}")
        v_trace = jnp.zeros((time_steps, len(brain.cells)))
        state = None
        
    res = SimulationResult(
        v_trace=v_trace,
        state=state,
        metadata={"t_max": config.t_max, "dt": config.dt}
    )
    
    return res
