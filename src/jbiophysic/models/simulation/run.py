# src/jbiophysic/models/simulation/run.py
import jaxley as jx
from typing import Dict, Any, Optional
from jbiophysic.common.types.simulation import SimulationConfig, SimulationResult
from jbiophysic.common.utils.logging import get_logger

logger = get_logger(__name__)

def run_simulation(
    brain: jx.Network, 
    config: SimulationConfig,
    params: Optional[Dict[str, Any]] = None
) -> SimulationResult:
    """
    Simulation runner for cortical models.
    Integration errors are intentionally surfaced to prevent silent failures.
    """
    logger.info(f"Running simulation for T={config.t_max}ms with dt={config.dt}ms")
    
    # We rely on Jaxley's internal error handling.
    # No silent zero-trace fallbacks are implemented here.
    v_trace, currents, state = jx.integrate(
        brain, 
        t_max=config.t_max, 
        dt=config.dt
    )
    logger.info("Integration successful.")
        
    return SimulationResult(
        v_trace=v_trace,
        currents=currents,
        state=state,
        metadata={
            "t_max": config.t_max, 
            "dt": config.dt,
            "seed": config.seed
        }
    )
