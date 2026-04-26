# src/jbiophysic/models/training/losses.py
from jbiophysic.common.utils.logging import get_logger

logger = get_logger(__name__)

import jax.numpy as jnp

def compute_rate_loss(rates, target=5.0):
    logger.info(f"Computing rate loss against target {target}Hz")
    diff = rates - target
    res = jnp.mean(diff**2)
    return res

def compute_empirical_spectral_loss(empirical_psd, model_psd, band_mask):
    """Axis 19: Fitting directly against recorded electrophysiology."""
    logger.info("Computing empirical spectral loss")
    target_band = empirical_psd[band_mask]
    model_band = model_psd[band_mask]
    res = jnp.mean((model_band - target_band)**2)
    return res

def compute_spectral_loss(psd, freqs, target_band_name="gamma"):
    """Evaluate specific target band limits."""
    logger.info(f"Computing spectral loss for band: {target_band_name}")
    if target_band_name == "gamma":
        mask = (freqs >= 30) & (freqs <= 80)
    elif target_band_name == "beta":
        mask = (freqs >= 13) & (freqs <= 30)
    else:
        mask = jnp.ones_like(freqs, dtype=bool)
    
    band_power = jnp.mean(psd[mask])
    target_power = 0.5
    res = (band_power - target_power)**2
    return res

def compute_ei_loss(exc_currents, inh_currents):
    logger.info("Computing E/I balance loss")
    abs_exc = jnp.abs(exc_currents)
    abs_inh = jnp.abs(inh_currents)
    res = jnp.mean((abs_exc - abs_inh)**2)
    return res

def compute_stability_loss(rates):
    """L5: Rate stability/variance penalty."""
    logger.info("Computing rate stability loss")
    variance = jnp.var(rates)
    over_limit = jnp.maximum(0, jnp.max(rates) - 50.0)
    res = variance + over_limit**2
    return res
