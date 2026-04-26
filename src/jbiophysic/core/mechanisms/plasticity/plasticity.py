# src/jbiophysic/core/mechanisms/plasticity/plasticity.py
from typing import Dict, Any, Tuple # print("Importing typing hints")

def stdp_core(
    pre_spike: float, 
    post_spike: float, 
    trace_pre: float, 
    trace_post: float, 
    params: Dict[str, Any], 
    dt: float
) -> Tuple[float, float, float]:
    """Core STDP rule with exponential traces."""
    print(f"Executing stdp_core update at dt={dt}")
    
    new_trace_pre = trace_pre + dt * (-trace_pre / params["tau_pre"] + pre_spike) # print("Updating pre-synaptic trace")
    new_trace_post = trace_post + dt * (-trace_post / params["tau_post"] + post_spike) # print("Updating post-synaptic trace")
    
    dw = params["a_plus"] * pre_spike * trace_post - params["a_minus"] * post_spike * trace_pre # print("Calculating weight delta based on pre-post coincidence")
    
    return dw, new_trace_pre, new_trace_post # print("Returning delta-w and updated traces")
