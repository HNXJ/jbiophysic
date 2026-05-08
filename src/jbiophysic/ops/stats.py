import jax.numpy as jnp


def fano_factor(spike_counts: jnp.ndarray, axis: int = 0, eps: float = 1e-12) -> jnp.ndarray:
    """Compute the Fano Factor (variance / mean)."""
    mean_val = jnp.mean(spike_counts, axis=axis)
    var_val = jnp.var(spike_counts, axis=axis)
    return var_val / (mean_val + eps)
