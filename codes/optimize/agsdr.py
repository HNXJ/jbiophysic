# codes/optimize/agsdr.py
import jax
import jax.numpy as jnp
from typing import Dict, Any

def compute_rate_loss(rates, target=5.0):
    return jnp.mean((rates - target)**2)

def compute_empirical_spectral_loss(empirical_psd, model_psd, band_mask):
    """Axis 19: Fitting directly against recorded electrophysiology and drug perturbation records."""
    target_band = empirical_psd[band_mask]
    model_band = model_psd[band_mask]
    return jnp.mean((model_band - target_band)**2)

def compute_spectral_loss(psd, freqs, target_band_name="gamma"):
    """Fixed contradictory logic: Evaluate specific target band limits instead of single peak."""
    if target_band_name == "gamma":
        mask = (freqs >= 30) & (freqs <= 80)
    elif target_band_name == "beta":
        mask = (freqs >= 13) & (freqs <= 30)
    else:
        mask = jnp.ones_like(freqs, dtype=bool)
    
    # Calculate power strictly in that band compared to a target scalar (if no empirical trace available)
    band_power = jnp.mean(psd[mask])
    target_power = 0.5 # Example fixed target
    return (band_power - target_power)**2

def compute_ei_loss(exc_currents, inh_currents):
    return jnp.mean((jnp.abs(exc_currents) - jnp.abs(inh_currents))**2)

def compute_stability_loss(rates):
    """L5: Rate stability/variance penalty."""
    return jnp.var(rates) + jnp.maximum(0, jnp.max(rates) - 50.0)**2

class AGSDR:
    """
    Adaptive Gradient Synaptic Drift Regularization (Axis 11/12).
    """
    def __init__(self, eta=0.001, lambdas=None):
        self.eta = eta
        self.lambdas = lambdas or {
            "rate": 1.0, "gamma": 0.5, "beta": 0.5, "ei": 0.5, "stability": 0.2
        }

    def compute_total_loss(self, state: Dict[str, Any], empirical_target: Dict[str, Any] = None):
        l_rate = compute_rate_loss(state["rates"])
        
        # Axis 19: Empirical Data Fitting Route
        if empirical_target is not None:
            l_gamma = compute_empirical_spectral_loss(empirical_target["psd"], state["psd"], empirical_target["gamma_mask"])
            l_beta = compute_empirical_spectral_loss(empirical_target["psd"], state["psd"], empirical_target["beta_mask"])
        else:
            # Axis 18: Evaluating band power instead of a single peak
            l_gamma = compute_spectral_loss(state["psd"], state["freqs"], target_band_name="gamma")
            l_beta = compute_spectral_loss(state["psd"], state["freqs"], target_band_name="beta")
            
        l_ei = compute_ei_loss(state["exc"], state["inh"])
        l_stab = compute_stability_loss(state["rates"])
        
        # Axis 19: Pharmacodynamic Perturbation Penalty (e.g. Ketamine blocking NMDAr)
        l_pharma = 0.0
        if "drug_target" in state:
            occupancy = state.get("receptor_occupancy", 0.0)
            target_occupancy = empirical_target.get("target_occupancy", 0.5) if empirical_target else 0.5
            l_pharma = (occupancy - target_occupancy)**2
        
        return (self.lambdas["rate"] * l_rate + 
                self.lambdas["gamma"] * l_gamma + 
                self.lambdas["beta"] * l_beta +
                self.lambdas["ei"] * l_ei +
                self.lambdas["stability"] * l_stab + 
                0.5 * l_pharma)

    def update_weights(self, weights, grad, g_clip=5.0, g_max=10.0):
        # Axis 14: Physiological bounds-based clipping
        drift = -self.eta * jnp.clip(grad, -g_clip, g_clip)
        return jnp.clip(weights + drift, 0.0, g_max)

