# src/jbiophysic/core/mechanisms/modulators/modulation.py
import jax.numpy as jnp
from typing import Dict, Any, Tuple

def get_experimental_schedule(phase: str) -> Dict[str, Any]:
    """
    Axis 11: Retrieves experimental schedule for phase (calibration, training, testing).
    """
    schedules = {
        "calibration": {"da": 0.5, "ach": 0.1, "stdp_on": False},
        "training":    {"da": 0.2, "ach": 0.8, "stdp_on": True},
        "testing":     {"da": 0.8, "ach": 0.2, "stdp_on": False}
    }
    return schedules.get(phase, schedules["calibration"])

def compute_functional_modulation(da: float, ach: float) -> Dict[str, float]:
    """
    Axis 11: Computes functional modulation gains for DA and ACh levels.
    """
    return {
        "precision": da * 2.0,
        "nmda_gain": 1.0 + (da * 0.5),
        "stdp_gain": ach * 1.5,
        "ff_gain": 1.0 + (ach * 0.8),
        "fb_gain": 1.0 - (ach * 0.5)
    }

def apply_modulation(state: Any, phase: str) -> Tuple[Dict[str, float], bool]:
    """
    Applies phase-specific modulation to the system state.
    """
    sched = get_experimental_schedule(phase)
    params = compute_functional_modulation(sched["da"], sched["ach"])
    return params, sched["stdp_on"]
