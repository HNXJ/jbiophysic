# src/jbiophysic/common/types/simulation.py


from dataclasses import dataclass
from typing import Any

import jax.numpy as jnp

from jbiophysic.common.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class SimulationConfig:
    dt: float = 0.025
    t_max: float = 500.0
    seed: int = 123


@dataclass
class SimulationResult:
    v_trace: jnp.ndarray
    currents: jnp.ndarray | None = None
    spikes: jnp.ndarray | None = None
    state: Any | None = None
    metadata: dict[str, Any] | None = None
