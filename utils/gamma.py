import jax.numpy as jnp
from typing import Dict, Any, List

def gamma_init():
    """Initialize a Gamma Trace context (Axis 4)."""
    return {
        "trace": [],
        "step": 0,
        "mode": "full" # "full", "summary", "off"
    }

def gamma_log(gamma: Dict[str, Any], tag: str, data: Any):
    """
    Structured logging for Gamma Trace.
    
    In 'full' mode: Logs everything.
    In 'summary' mode: Logs mean/std if data is a tensor.
    In 'off' mode: Does nothing.
    """
    if gamma["mode"] == "off":
        return gamma
    
    if gamma["mode"] == "summary" and hasattr(data, "mean"):
        data = {
            "mean": jnp.mean(data),
            "std": jnp.std(data)
        }
    
    gamma["trace"].append({
        "step": gamma["step"],
        "tag": tag,
        "data": data
    })
    
    return gamma

def step_with_gamma(state, params, gamma, dt, step_fn):
    """
    Meta-wrapper (Axis 4) for simulation steps:
    - Logs input state.
    - Executes biophysics.
    - Logs updates.
    """
    gamma = gamma_log(gamma, "entry_state", state)
    
    # 1. Execute the biophysical step (e.g. neuron_step or predictive_step)
    # This matches: Traceable Execution
    new_state, logs = step_fn(state, params, dt)
    
    # 2. Log internal variables (currents, updates) from the step_fn
    for tag, val in logs.items():
        gamma = gamma_log(gamma, tag, val)
    
    gamma["step"] += 1
    
    return new_state, gamma
