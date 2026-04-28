import jax.numpy as jnp
import numpy as np
from typing import Dict, Any
from jbiophysic.models.builders.v1_column_model import build_v1_column_jaxley
from jbiophysic.models.training.stability_gatekeeper import evaluate_stability
from jbiophysic.common.utils.logging import get_logger
import jaxley as jx
from scipy.signal import welch

logger = get_logger(__name__)

def compute_gamma_ratio(v_trace: jnp.ndarray, fs: float, onset_idx: int) -> float:
    """
    Computes ratio = evoked_gamma_power / baseline_gamma_power.
    Uses PSD computed from explicit time windows.
    v_trace shape: (num_cells, num_time_steps)
    """
    # Use population mean as a simple LFP proxy for the network
    lfp_proxy = jnp.mean(v_trace, axis=0)
    
    baseline_trace = lfp_proxy[:onset_idx]
    evoked_trace = lfp_proxy[onset_idx:]
    
    # Using scipy.signal.welch for PSD
    # fs is sampling freq
    fb, psd_b = welch(np.array(baseline_trace), fs, nperseg=min(256, len(baseline_trace)))
    fe, psd_e = welch(np.array(evoked_trace), fs, nperseg=min(256, len(evoked_trace)))
    
    gamma_band = (30, 80)
    
    # Mean power in gamma band
    b_gamma = np.mean(psd_b[(fb >= gamma_band[0]) & (fb <= gamma_band[1])])
    e_gamma = np.mean(psd_e[(fe >= gamma_band[0]) & (fe <= gamma_band[1])])
    
    # Avoid div by zero
    b_gamma = max(b_gamma, 1e-12)
    ratio = float(e_gamma / b_gamma)
    
    return ratio

def run_v1_gamma_bridge_payload(payload_dict: dict) -> Dict[str, Any]:
    """
    1. Build network with payload params using built-in HH.
    2. Integrate (Baseline + Evoked).
    3. Compute real gamma ratio from traces.
    4. Evaluate stability on real state tensors.
    5. Return empirically derived bundle.
    """
    logger.info("Executing real V1 Gamma Bridge Payload with built-in HH.")
    
    params = payload_dict.get("params", {})
    t_max = 300.0
    dt = 0.025
    fs = 1000.0 / dt
    onset_ms = 100.0
    onset_idx = int(onset_ms / dt)
    
    # 1. Build
    net = build_v1_column_jaxley(params)
    
    # 2. Simulate
    logger.info(f"Integrating Jaxley network for {t_max}ms")
    v_trace = jx.integrate(net, t_max=t_max, delta_t=dt)
    
    # 3. Analyze
    gamma_ratio = compute_gamma_ratio(v_trace, fs, onset_idx)
    
    # 4. Stability
    stability = evaluate_stability(v_trace, dt)
    
    if not stability["passed"]:
        logger.warning(f"Stability check failed: {stability['reason']}")
        return {
            "status": "rejected",
            "gamma_ratio": 0.0,
            "stability": stability
        }
        
    logger.info(f"Simulation success. Gamma Ratio: {gamma_ratio:.3f}")
    
    # 5. Bundle
    return {
        "status": "success",
        "gamma_ratio": gamma_ratio,
        "stability": stability,
        "metadata": {
            "t_max": t_max,
            "dt": dt,
            "pv_gain": params.get("pv_gain", 1.0),
            "v_max": stability.get("v_max"),
            "v_min": stability.get("v_min")
        }
    }
