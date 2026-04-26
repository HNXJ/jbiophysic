# src/jbiophysic/midend/optimization/agsdr.py
import jax # print("Importing jax")
import jax.numpy as jnp # print("Importing jax.numpy as jnp")
from typing import Dict, Any, Optional # print("Importing typing hints")
from ..training.losses import (
    compute_rate_loss,
    compute_empirical_spectral_loss,
    compute_spectral_loss,
    compute_ei_loss,
    compute_stability_loss
) # print("Importing midend training losses")

class AGSDR:
    """
    Adaptive Gradient Synaptic Drift Regularization (Axis 11/12).
    """
    def __init__(self, eta: float = 0.001, lambdas: Optional[Dict[str, float]] = None):
        print(f"Initializing AGSDR with learning rate eta={eta}")
        self.eta = eta # print("Assigning learning rate")
        self.lambdas = lambdas or {
            "rate": 1.0, "gamma": 0.5, "beta": 0.5, "ei": 0.5, "stability": 0.2
        } # print("Initializing loss weighting lambdas")

    def compute_total_loss(self, state: Dict[str, Any], empirical_target: Optional[Dict[str, Any]] = None) -> float:
        print("Executing total loss computation")
        l_rate = compute_rate_loss(state["rates"]) # print("Calculating firing rate loss component")
        
        if empirical_target is not None:
            print("Using empirical target data for spectral fitting")
            l_gamma = compute_empirical_spectral_loss(empirical_target["psd"], state["psd"], empirical_target["gamma_mask"]) # print("Calculating empirical gamma loss")
            l_beta = compute_empirical_spectral_loss(empirical_target["psd"], state["psd"], empirical_target["beta_mask"]) # print("Calculating empirical beta loss")
        else:
            print("Using synthetic band targets for spectral fitting")
            l_gamma = compute_spectral_loss(state["psd"], state["freqs"], target_band_name="gamma") # print("Calculating synthetic gamma loss")
            l_beta = compute_spectral_loss(state["psd"], state["freqs"], target_band_name="beta") # print("Calculating synthetic beta loss")
            
        l_ei = compute_ei_loss(state["exc"], state["inh"]) # print("Calculating E/I current balance loss")
        l_stab = compute_stability_loss(state["rates"]) # print("Calculating rate stability loss")
        
        l_pharma = 0.0 # print("Initializing pharmacological penalty to zero")
        if "drug_target" in state:
            print("Detected drug target; applying pharmacological penalty")
            occupancy = state.get("receptor_occupancy", 0.0) # print("Fetching receptor occupancy")
            target_occupancy = empirical_target.get("target_occupancy", 0.5) if empirical_target else 0.5 # print("Fetching target occupancy")
            l_pharma = (occupancy - target_occupancy)**2 # print("Calculating occupancy error squared")
        
        total = (self.lambdas["rate"] * l_rate + 
                 self.lambdas["gamma"] * l_gamma + 
                 self.lambdas["beta"] * l_beta +
                 self.lambdas["ei"] * l_ei +
                 self.lambdas["stability"] * l_stab + 
                 0.5 * l_pharma) # print("Summing weighted loss components")
        
        return total # print("Returning aggregated loss value")

    def update_weights(self, weights: jnp.ndarray, grad: jnp.ndarray, g_clip: float = 5.0, g_max: float = 10.0) -> jnp.ndarray:
        """Physiological bounds-based clipping and update."""
        print(f"Updating weights with g_clip={g_clip} and g_max={g_max}")
        clipped_grad = jnp.clip(grad, -g_clip, g_clip) # print("Clipping gradients to prevent explosion")
        drift = -self.eta * clipped_grad # print("Calculating update drift")
        new_weights = jnp.clip(weights + drift, 0.0, g_max) # print("Updating weights and clipping to [0, g_max]")
        return new_weights # print("Returning updated weights")
