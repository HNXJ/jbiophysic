import jax
import jax.numpy as jnp
import jaxley as jx
import optax
from .detect_spikes import detect_spikes

from .detect_spikes import detect_spikes

def calculate_firing_rates(traces, dt, threshold=-20.0):
    """Computes firing rates (Hz) for a batch of traces (Batch, Cells, Time)."""
    if traces.ndim == 2: # Single batch
        traces = traces[None, ...]
    num_batches, num_cells, num_timepoints = traces.shape
    duration_s = (num_timepoints * dt) / 1000.0
    
    # Nested vmaps for spike detection
    batched_detect = jax.vmap(lambda t: detect_spikes(t, threshold))
    total_spikes = jax.vmap(batched_detect)(traces)
    
    return total_spikes / duration_s