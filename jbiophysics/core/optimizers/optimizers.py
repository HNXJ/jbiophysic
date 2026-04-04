import jax
import jax.numpy as jnp
import optax
from flax.struct import dataclass
from typing import Any, Callable, Optional, Tuple, List
import numpy as np

# Import specialized optimizers
from jbiophysics.core.optimizers.SDR import SDR, SDRState
from jbiophysics.core.optimizers.GSDR import GSDR, GSDRState
from jbiophysics.core.optimizers.AGSDR import AGSDR

# --- Transforms ---

class ClampTransform:
    def __init__(self, lower, upper):
        self.lower = lower
        self.upper = upper
    def forward(self, x):
        return jnp.clip(x, self.lower, self.upper)

# --- Analysis Tools ---

def compute_kappa(spike_matrix: jnp.ndarray, fs: float = 10000.0, bin_size_ms: float = 5.0) -> float:
    """
    Computes Fleiss' Kappa for a population of neurons to quantify synchrony.
    Targeting [-0.1, 0.1] for physiological asynchrony.
    """
    # 1. Binning
    bin_size_samples = int(bin_size_ms * fs / 1000.0)
    num_bins = spike_matrix.shape[1] // bin_size_samples
    
    # Reshape and sum to get counts per bin
    # shape: (cells, bins, samples_per_bin)
    binned = spike_matrix[:, :num_bins * bin_size_samples].reshape(
        spike_matrix.shape[0], num_bins, bin_size_samples
    ).sum(axis=2)
    binned = (binned > 0).astype(float) # Binary: fired or not in bin
    
    # 2. Fleiss' Kappa Math
    N, k = binned.shape # N cells, k bins
    
    # Degree of agreement for each bin
    # n_ij is count of cells spiking in bin j
    n_spiking = binned.sum(axis=0)
    P_i = (n_spiking**2 - n_spiking + (N - n_spiking)**2 - (N - n_spiking)) / (N * (N - 1 + 1e-12))
    P_bar = P_i.mean()
    
    # Expected agreement
    P_mean = binned.mean()
    P_e = P_mean**2 + (1 - P_mean)**2
    
    kappa = (P_bar - P_e) / (1 - P_e + 1e-12)
    
    # Handle edge cases (all spike or none spike)
    is_invalid = (P_mean <= 0.0) | (P_mean >= 1.0) | (N < 2) | (num_bins == 0)
    kappa = jnp.where(is_invalid, 0.0, kappa)
    
    return kappa
