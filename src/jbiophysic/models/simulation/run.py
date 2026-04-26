# src/jbiophysic/models/simulation/run.py
import jax
import jax.numpy as jnp
import jaxley as jx
from typing import Dict, Any, Tuple, Optional
from jbiophysic.common.types.simulation import SimulationConfig, SimulationResult
from jbiophysic.common.utils.logging import get_logger

logger = get_logger(__name__)

def run_simulation(
    brain: jx.Network, 
    config: SimulationConfig,
    params: Optional[Dict[str, Any]] = None
) -> SimulationResult:
    """
    Axis 18: Canonical models simulation runner.
    """
    logger.info(f"Running simulation for T={config.t_max}ms with dt={config.dt}ms")
    
    # 1. Apply stimuli (Placeholder logic)
    # time_steps = int(config.t_max / config.dt)
    
    logger.info("Starting Jaxley integration...")
    # Jaxley.integrate returns (v_trace, currents, states)
    # Recording currents is essential for biophysical analysis (LFP, E/I balance).
    v_trace, currents, state = jx.integrate(
        brain, 
        t_max=config.t_max, 
        dt=config.dt
    )
    logger.info("Integration successful.")
        
    res = SimulationResult(
        v_trace=v_trace,
        currents=currents,
        state=state,
        metadata={
            "t_max": config.t_max, 
            "dt": config.dt,
            "seed": config.seed
        }
    )
    
    return res
