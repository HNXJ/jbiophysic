import jax.numpy as jnp
from typing import Dict, Any

def evaluate_stability(v_trace: jnp.ndarray, dt: float) -> Dict[str, Any]:
    """
    Analyzes simulated voltage traces for numerical stability.
    Returns a dict with 'passed' boolean and reason if failed.
    """
    # 1. Check for NaNs or Infs
    if jnp.any(jnp.isnan(v_trace)):
        return {"passed": False, "reason": "NaN detected in voltage traces"}
    
    if jnp.any(jnp.isinf(v_trace)):
        return {"passed": False, "reason": "Inf detected in voltage traces"}
    
    # 2. Check for physiological bounds
    # Hodgkin-Huxley voltages should stay within roughly [-100, 50] mV.
    # We use [-120, 100] as a safe "unstable" boundary.
    v_max = jnp.max(v_trace)
    v_min = jnp.min(v_trace)
    
    if v_max > 100.0:
        return {"passed": False, "reason": f"Voltage blowout: max(V)={v_max:.1f}mV"}
    
    if v_min < -120.0:
        return {"passed": False, "reason": f"Voltage blowout: min(V)={v_min:.1f}mV"}
        
    # 3. Check for flatlines (if desired, but zero noise baseline can be flat)
    v_std = jnp.std(v_trace)
    if v_std < 1e-6:
        # Not strictly a failure, but worth noting
        pass
        
    return {
        "passed": True, 
        "v_max": float(v_max), 
        "v_min": float(v_min),
        "v_std": float(v_std)
    }
