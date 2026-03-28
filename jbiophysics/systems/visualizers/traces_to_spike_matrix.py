import jax
import jax.numpy as jnp
import jaxley as jx
import optax

def traces_to_spike_matrix(traces: np.ndarray, threshold: float = -20.0) -> np.ndarray:
    """Converts voltage traces to a binary spike matrix."""
    num_neurons, num_steps = traces.shape
    spike_matrix = np.zeros_like(traces)
    for i in range(num_neurons):
        spikes = (traces[i, :-1] < threshold) & (traces[i, 1:] >= threshold)
        spike_matrix[i, 1:][spikes] = 1.0
    return spike_matrix