# src/jbiophysic/core/math/predictive.py
import jax.numpy as jnp

def predictive_step(error: float, prediction: float, precision: float) -> float:
    """
    Axis 11: Precision-weighted prediction error calculation.
    Computes the weighted mismatch between expected and actual sensory input.
    """
    return precision * (error - prediction)
