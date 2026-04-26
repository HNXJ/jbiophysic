# src/jbiophysic/core/mechanisms/plasticity/plasticity.py
import jax.numpy as jnp
from typing import Dict, Any, Tuple

def stdp_core(
    dt: float, 
    pre_spike: float, 
    post_spike: float, 
    trace_pre: float, 
    trace_post: float, 
    params: Dict[str, float]
) -> Tuple[float, float, float]:
    """
    Axis 14: Pure JAX STDP kernel for synaptic weight updates.
    """
    new_trace_pre = trace_pre + dt * (-trace_pre / params["tau_pre"] + pre_spike)
    new_trace_post = trace_post + dt * (-trace_post / params["tau_post"] + post_spike)
    
    # Weight change: Pre before post -> Potentiation (a_plus); Post before pre -> Depression (a_minus)
    # LTP: post-spike triggered by pre-trace
    # LTD: pre-spike triggered by post-trace
    dw = params["a_plus"] * post_spike * trace_pre - params["a_minus"] * pre_spike * trace_post
    
    return dw, new_trace_pre, new_trace_post
