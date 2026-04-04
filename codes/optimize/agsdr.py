# codes/optimize/agsdr.py
import jax
import jax.numpy as jnp
from typing import Dict, Any

def compute_rate_loss(rates, target=5.0):
    return jnp.mean((rates - target)**2)

def compute_spectral_loss(lfp, target_freq=40.0, fs=1000.0):
    freqs = jnp.fft.rfftfreq(lfp.shape[-1], d=1/fs)
    fft_mag = jnp.abs(jnp.fft.rfft(lfp, axis=-1))
    peak_freq = freqs[jnp.argmax(fft_mag, axis=-1)]
    return jnp.mean((peak_freq - target_freq)**2)

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

    def compute_total_loss(self, state: Dict[str, Any]):
        l_rate = compute_rate_loss(state["rates"])
        l_gamma = compute_spectral_loss(state["lfp"], target_freq=40.0)
        l_beta = compute_spectral_loss(state["lfp"], target_freq=20.0)
        l_ei = compute_ei_loss(state["exc"], state["inh"])
        l_stab = compute_stability_loss(state["rates"])
        
        return (self.lambdas["rate"] * l_rate + 
                self.lambdas["gamma"] * l_gamma + 
                self.lambdas["beta"] * l_beta +
                self.lambdas["ei"] * l_ei +
                self.lambdas["stability"] * l_stab)

    def update_weights(self, weights, grad, g_max=10.0):
        drift = -self.eta * jnp.clip(grad, -1.0, 1.0)
        return jnp.clip(weights + drift, 0.0, g_max)
