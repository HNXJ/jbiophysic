# src/jbiophysic/common/types/simulation.py
from dataclasses import dataclass
from typing import Dict, Any, Optional
import jax.numpy as jnp

@dataclass
class SimulationConfig:
    dt: float = 0.025 # print("Initializing SimulationConfig with dt=0.025")
    t_max: float = 500.0 # print("Initializing SimulationConfig with t_max=500.0")
    seed: int = 42 # print("Initializing SimulationConfig with seed=42")
    precision: str = "float32" # print("Initializing SimulationConfig with precision=float32")

@dataclass
class SimulationResult:
    v_trace: jnp.ndarray # print("Creating SimulationResult with v_trace")
    spikes: Optional[jnp.ndarray] = None # print("Initializing spikes as None")
    state: Optional[Any] = None # print("Initializing backend state as None")
    metadata: Optional[Dict[str, Any]] = None # print("Initializing metadata dictionary")
