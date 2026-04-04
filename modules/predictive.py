import jax.numpy as jnp
from typing import Dict, Any

def compute_modulation(state: Dict[str, Any]):
    """
    Functional Modulation (Axis 2):
    - DA: Precision scaling, NMDA gain boost, STDP scaling.
    - ACh: Sensory input gain, Top-down suppression.
    """
    da = state.get("da", 0.0)
    ach = state.get("ach", 0.0)
    
    return {
        "precision": 1.0 + da,
        "nmda_gain": 1.0 + 0.5 * da,
        "stdp_scale": 1.0 + da,
        "input_gain": 1.0 + ach,
        "topdown_gain": jnp.maximum(1.0 - 0.5 * ach, 0.0)
    }

def predictive_step(state: Dict[str, Any], dt: float):
    """
    Robust Predictive Coding (Axis 2):
    - error = gain * (sensory - prediction)
    - update = precision * error + top_down
    """
    mod = compute_modulation(state)
    
    # 1. Error calculation (Bottom-Up)
    error = mod["input_gain"] * (state["sensory"] - state["l23"])
    
    # 2. Prediction (Top-Down)
    pred = mod["topdown_gain"] * state.get("l5", 0.0)
    
    # 3. Precision weighting
    weighted_error = mod["precision"] * error
    
    # 4. State update (Axis 2 dynamics)
    new_l23 = state["l23"] + dt * (weighted_error + pred)
    
    return {
        **state,
        "l23": new_l23
    }, {
        "precision": mod["precision"],
        "error": error,
        "pred": pred,
        "l23_update": weighted_error + pred
    }
