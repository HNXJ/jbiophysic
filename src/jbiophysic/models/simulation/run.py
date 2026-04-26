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
    time_steps = int(config.t_max / config.dt)
    
    try:
        logger.info("Starting Jaxley integration...")
        # Jaxley.integrate returns (v_trace, currents, states)
        # v_trace shape is typically (n_compartments, n_time_steps)
        v_trace, _, state = jx.integrate(
            brain, 
            t_max=config.t_max, 
            dt=config.dt
        )
        logger.info("Integration successful.")
    except Exception as e:
        logger.error(f"CRITICAL: Jaxley integration failed: {str(e)}")
        # Fallback must match the [neurons, time] orientation
        v_trace = jnp.zeros((len(brain.cells), time_steps))
        state = None
        
    res = SimulationResult(
        v_trace=v_trace,
        state=state,
        metadata={"t_max": config.t_max, "dt": config.dt}
    )
    
    return res
