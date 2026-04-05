# codes/plasticity.py
def stdp_core(pre_spike, post_spike, trace_pre, trace_post, params, dt):
    """Core STDP rule with exponential traces."""
    trace_pre = trace_pre + dt * (-trace_pre / params["tau_pre"] + pre_spike)
    trace_post = trace_post + dt * (-trace_post / params["tau_post"] + post_spike)
    dw = params["a_plus"] * pre_spike * trace_post - params["a_minus"] * post_spike * trace_pre
    return dw, trace_pre, trace_post

# codes/predictive.py
def predictive_step(error, prediction, precision):
    """Bridge for precision-weighted predictive coding."""
    return (error - prediction) * precision

# codes/hierarchy.py
import jax.numpy as jnp
def build_11_area_hierarchy():
    """Area mapping: V1/V2 (Low) -> V4-FST (Mid) -> V3A-PFC (High)."""
    return {
        "low": ["V1", "V2"],
        "mid": ["V4", "MT", "MST", "TEO", "FST"],
        "high": ["V3A", "V3D", "FEF", "PFC"]
    }

# codes/modulation.py
def compute_modulation(da, ach):
    """Neuromodulatory scaling of precision and gain."""
    return {
        "da_precision": 1.0 + da,
        "ach_gain": 1.0 + ach,
        "stdp_delta": 0.01 * (1.0 + da)
    }
