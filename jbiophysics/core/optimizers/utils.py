import jax
import jax.numpy as jnp
import optax
from typing import Any

def apply_spatial_smoothing(x: jnp.ndarray, k_size: int = None) -> jnp.ndarray:
    """
    Applies a uniform smoothing kernel to a parameter array.
    Useful for biophysical parameters that vary smoothly across space.
    """
    if x.ndim == 2:
        n, m = x.shape
        kn = k_size if k_size else jnp.maximum(1, jnp.sqrt(n).astype(int))
        km = k_size if k_size else jnp.maximum(1, jnp.sqrt(m).astype(int))
        kernel = jnp.ones((kn, km)) / (kn * km)
        # Using a valid pad or same? 'same' is usually better for net params
        return jax.scipy.signal.convolve2d(x, kernel, mode='same')
    elif x.ndim == 1:
        n = x.shape[0]
        k = k_size if k_size else jnp.maximum(1, jnp.sqrt(n).astype(int))
        kernel = jnp.ones((k,)) / k
        return jnp.convolve(x, kernel, mode='same')
    return x

def calculate_layerwise_variance(updates: Any) -> Any:
    """Returns a tree of variances for each leaf in the updates."""
    return jax.tree.map(lambda x: jnp.var(x) + 1e-12, updates)

def success_expansion(time_since_last_success: jnp.ndarray, tau: float = 10.0) -> jnp.ndarray:
    """
    Sigmoidal expansion factor for exploration.
    Drives exploration magnitude based on how long we have been 'stuck'.
    """
    return (time_since_last_success**2) * (1.0 - jnp.exp(-time_since_last_success / tau))
