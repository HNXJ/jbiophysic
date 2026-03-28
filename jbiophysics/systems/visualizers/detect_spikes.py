import jax
import jax.numpy as jnp
import jaxley as jx
import optax

def detect_spikes(neuron_trace, threshold=-20.0):
    """Detects upward crossings of a threshold."""
    # Exclude first point if it's V_init
    data = neuron_trace[1:]
    spikes = (data[:-1] < threshold) & (data[1:] >= threshold)
    return jnp.sum(spikes)