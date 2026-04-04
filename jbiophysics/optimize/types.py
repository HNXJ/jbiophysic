import jax
import jax.numpy as jnp
from typing import Any
from flax.struct import dataclass

@dataclass
class GSDRState:
    """State for GSDR and AGSDR optimizers."""
    inner_state: Any
    params_opt: Any
    inner_state_opt: Any
    loss_opt: float
    a: float
    a_opt: float
    lambda_d: float
    step_count: int
    consecutive_unchanged_epochs: int
    last_optimal_change_step: int
    var_sup_ema: float = 1.0
    var_unsup_ema: float = 1.0
