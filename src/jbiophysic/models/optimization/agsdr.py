# src/jbiophysic/models/optimization/agsdr.py
from jbiophysic.common.utils.logging import get_logger

logger = get_logger(__name__)

import jax
import jax.numpy as jnp
from typing import Dict, Any, Optional
from jbiophysic.models.training.losses import (
    compute_rate_loss,
    compute_empirical_spectral_loss,
    compute_spectral_loss,
    compute_ei_loss,
    compute_stability_loss
)

class AGSDR:
    """
    Adaptive Gradient Synaptic Drift Regularization (Axis 11/12).
    """
    def __init__(self, eta: float = 0.001, lambdas: Optional[Dict[str, float]] = None):
        logger.info(f"Initializing AGSDR with learning rate eta={eta}")
        self.eta = eta
        self.lambdas = lambdas or {
            "rate": 1.0, "gamma": 0.5, "beta": 0.5, "ei": 0.5, "stability": 0.2
        }

    def compute_total_loss(self, state: Dict[str, Any], empirical_target: Optional[Dict[str, Any]] = None) -> float:
        l_rate = compute_rate_loss(state["rates"])
        
        if empirical_target is not None:
            l_gamma = compute_empirical_spectral_loss(empirical_target["psd"], state["psd"], empirical_target["gamma_mask"])
            l_beta = compute_empirical_spectral_loss(empirical_target["psd"], state["psd"], empirical_target["beta_mask"])
        else:
            l_gamma = compute_spectral_loss(state["psd"], state["freqs"], target_band_name="gamma")
            l_beta = compute_spectral_loss(state["psd"], state["freqs"], target_band_name="beta")
            
        l_ei = compute_ei_loss(state["exc"], state["inh"])
        l_stab = compute_stability_loss(state["rates"])
        
        # JAX-pure drug target check using select
        has_drug = state.get("has_drug_target", 0.0) # Explicit flag
        occupancy = state.get("receptor_occupancy", 0.0)
        target_occupancy = empirical_target.get("target_occupancy", 0.5) if empirical_target else 0.5
        l_pharma_val = (occupancy - target_occupancy)**2
        l_pharma = jax.lax.select(has_drug > 0.5, l_pharma_val, 0.0)
        
        total = (self.lambdas["rate"] * l_rate + 
                 self.lambdas["gamma"] * l_gamma + 
                 self.lambdas["beta"] * l_beta +
                 self.lambdas["ei"] * l_ei +
                 self.lambdas["stability"] * l_stab + 
                 0.5 * l_pharma)
        
        return total

    def update_weights(self, weights: jnp.ndarray, grad: jnp.ndarray, g_clip: float = 5.0, g_max: float = 10.0) -> jnp.ndarray:
        """Physiological bounds-based clipping and update."""
        clipped_grad = jnp.clip(grad, -g_clip, g_clip)
        drift = -self.eta * clipped_grad
        new_weights = jnp.clip(weights + drift, 0.0, g_max)
        return new_weights
