import jax.numpy as jnp

def lfp_proxy(csd_or_sources: jnp.ndarray, weights: jnp.ndarray | None = None) -> jnp.ndarray:
    """A simple declared proxy for LFP from CSD or sources.
    
    This is not a validated LFP model.
    """
    if weights is None:
        return jnp.mean(csd_or_sources, axis=-1)
    return jnp.sum(csd_or_sources * weights, axis=-1)
