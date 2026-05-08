import jax.numpy as jnp


def per_neuron_firing_rate(spikes: jnp.ndarray, dt_ms: float) -> jnp.ndarray:
    """Compute the mean firing rate (Hz) for each neuron.
    
    Assumes spikes has shape [time, neurons].
    """
    duration_s = (spikes.shape[0] * dt_ms) / 1000.0
    return jnp.sum(spikes, axis=0) / duration_s

def firing_rate(spikes: jnp.ndarray, dt_ms: float) -> jnp.ndarray:
    """Compute the population mean firing rate (Hz)."""
    return jnp.mean(per_neuron_firing_rate(spikes, dt_ms))

def max_single_neuron_rate(spikes: jnp.ndarray, dt_ms: float) -> jnp.ndarray:
    """Compute the maximum firing rate (Hz) across all neurons."""
    return jnp.max(per_neuron_firing_rate(spikes, dt_ms))
