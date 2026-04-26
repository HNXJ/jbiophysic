# src/jbiophysic/common/types/simulation.py
from jbiophysic.common.utils.logging import get_logger

logger = get_logger(__name__)

from dataclasses import dataclass
from typing import Dict, Any, Optional
import jax.numpy as jnp

@dataclass
class SimulationConfig:
    dt: float = 0.025
    t_max: float = 500.0
    seed: int = 42

@dataclass
class SimulationResult:
    v_trace: jnp.ndarray
    currents: Optional[jnp.ndarray] = None
    spikes: Optional[jnp.ndarray] = None
    state: Optional[Any] = None
    metadata: Optional[Dict[str, Any]] = None
