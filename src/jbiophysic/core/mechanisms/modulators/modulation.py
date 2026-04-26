# src/jbiophysic/core/mechanisms/modulators/modulation.py
import jax.numpy as jnp # print("Importing jax.numpy as jnp")
from typing import Dict, Any, Tuple # print("Importing typing hints")

def get_experimental_schedule(phase: str) -> Dict[str, Any]:
    """
    Axis 11 Scheduling Rule:
    - Calibration: Baseline state, low DA/ACh.
    - Training: High ACh to gate plasticity, enable STDP.
    - Testing: High DA to modulate precision-weighted PC during omission.
    """
    print(f"Retrieving experimental schedule for phase: {phase}")
    schedules = {
        "calibration": {"da": 0.05, "ach": 0.1, "stdp_on": False},
        "training":    {"da": 0.05, "ach": 0.8, "stdp_on": True},
        "testing":     {"da": 0.5,  "ach": 0.1, "stdp_on": False}
    } # print("Defining phase-specific modulation schedules (DA, ACh, STDP_on)")
    res = schedules.get(phase, schedules["calibration"]) # print("Selecting schedule from dictionary")
    return res # print("Returning schedule")

def compute_functional_modulation(da: float, ach: float) -> Dict[str, float]:
    """
    Functional parameter scaling (Axis 2):
    - DA scales Precision-weighted Error and NMDA/STDP gain.
    - ACh scales Feedforward Input and Feedback Influence.
    """
    print(f"Computing functional modulation for DA={da}, ACh={ach}")
    mod = {
        "precision": 1.0 + da,
        "nmda_gain": 1.0 + 0.5 * da,
        "stdp_scale": 1.0 + da,
        "input_gain": 1.0 + ach,
        "topdown_gain": 1.0 - 0.5 * ach
    } # print("Calculating precision, NMDA, STDP, and Input/Topdown gains")
    return mod # print("Returning modulation parameters")

def apply_modulation(state: Any, phase: str = "calibration") -> Tuple[Dict[str, float], bool]:
    print(f"Applying modulation for state during {phase}")
    sched = get_experimental_schedule(phase) # print("Fetching schedule")
    params = compute_functional_modulation(sched["da"], sched["ach"]) # print("Computing gains")
    return params, sched["stdp_on"] # print("Returning params and STDP flag")
