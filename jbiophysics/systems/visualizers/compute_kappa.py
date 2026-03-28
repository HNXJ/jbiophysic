import jax
import jax.numpy as jnp
import jaxley as jx
import optax

def compute_kappa(spike_matrix, fs, bin_size_ms=5.0):
    """
    Computes Fleiss' Kappa for inter-neuron synchrony.
    Input: (Cells, Time) binary matrix.
    """
    N, T = spike_matrix.shape
    samples_per_bin = int(fs * (bin_size_ms / 1000.0)) or 1
    num_bins = T // samples_per_bin
    
    # Use a slice that is a multiple of samples_per_bin
    # We use dynamic_slice or just regular slicing if indices are known
    valid_T = num_bins * samples_per_bin
    binned_data = spike_matrix[:, :valid_T]
    
    # Reshape and take max across bin
    # Note: reshape with -1 or concrete values
    # To keep it JIT-friendly, we use the fact that valid_T is num_bins * samples_per_bin
    binned = jnp.max(binned_data.reshape(N, num_bins, samples_per_bin), axis=2)
    
    spikes_per_bin = binned.sum(axis=0)
    silences_per_bin = N - spikes_per_bin
    
    # P_i: relative observed agreement in bin i
    # Avoid division by zero if N < 2
    den_Pi = jnp.maximum(1.0, float(N * (N - 1)))
    P_i = (spikes_per_bin * (spikes_per_bin - 1) + silences_per_bin * (silences_per_bin - 1)) / den_Pi
    P_o = jnp.mean(P_i)
    
    # P_e: relative agreement expected by chance
    den_Pe = jnp.maximum(1.0, float(N * num_bins))
    p_spike = jnp.sum(spikes_per_bin) / den_Pe
    P_e = (p_spike**2) + ((1 - p_spike)**2)
    
    # Final Kappa
    return (P_o - P_e) / jnp.maximum(1e-6, 1.0 - P_e)