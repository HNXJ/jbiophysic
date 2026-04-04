# codes/modulation.py
import jax.numpy as jnp

def get_experimental_schedule(phase: str):
    """
    Axis 11 Scheduling Rule:
    - Calibration: Baseline state, low DA/ACh.
    - Training: High ACh to gate plasticity, enable STDP.
    - Testing: High DA to modulate precision-weighted PC during omission.
    """
    schedules = {
        "calibration": {"da": 0.05, "ach": 0.1, "stdp_on": False},
        "training":    {"da": 0.05, "ach": 0.8, "stdp_on": True},
        "testing":     {"da": 0.5,  "ach": 0.1, "stdp_on": False}
    }
    return schedules.get(phase, schedules["calibration"])

def compute_functional_modulation(da, ach):
    """
    Functional parameter scaling (Axis 2):
    - DA scales Precision-weighted Error and NMDA/STDP gain.
    - ACh scales Feedforward Input and Feedback Influence.
    """
    return {
        "precision": 1.0 + da,
        "nmda_gain": 1.0 + 0.5 * da,
        "stdp_scale": 1.0 + da,
        "input_gain": 1.0 + ach,
        "topdown_gain": 1.0 - 0.5 * ach
    }

def apply_modulation(state, phase="calibration"):
    sched = get_experimental_schedule(phase)
    params = compute_functional_modulation(sched["da"], sched["ach"])
    return params, sched["stdp_on"]
