"""
core/optimizers/omission_metrics.py

Three metrics for calibrating the omission model to empirical data
(Bastos et al. Chapter 2 targets):

  1. RSA  — Representational Similarity Analysis
             cosine distance between population spike-rate vectors
  2. Fleiss Kappa — burst-timing synchrony across laminar layers
  3. SSS  — Spectral Similarity Score, Welch PSD per canonical band
"""

import numpy as np
import jax.numpy as jnp
from typing import Dict, List, Optional, Tuple
from scipy.signal import welch
import scipy.stats


# ── 1. Representational Similarity Analysis ────────────────────────────────────

def population_rate_vector(
    spikes: Dict[int, np.ndarray],
    pop_indices: List[int],
    n_steps: int,
    bin_size: int = 200,
) -> np.ndarray:
    """
    Compute a binned firing-rate vector for a cell population.

    Args:
        spikes      – {cell_idx: spike_timestep_array}
        pop_indices – which cells belong to this population
        n_steps     – total simulation timesteps
        bin_size    – timesteps per bin

    Returns:
        rate_vec: shape (n_bins,) in spikes/bin
    """
    n_bins = n_steps // bin_size
    vec = np.zeros(n_bins, dtype=np.float64)
    for idx in pop_indices:
        if idx in spikes:
            bins = spikes[idx] // bin_size
            valid = bins[bins < n_bins]
            np.add.at(vec, valid, 1)
    return vec / max(len(pop_indices), 1)


def compute_rsa(
    spike_sim:    Dict[int, np.ndarray],
    spike_target: Dict[int, np.ndarray],
    pop_indices:  List[int],
    n_steps:      int,
    bin_size:     int = 200,
) -> float:
    """
    RSA: 1 - cosine_similarity between simulated and target rate vectors.
    Range [0, 2].  0 = perfect match.
    """
    v_sim = population_rate_vector(spike_sim, pop_indices, n_steps, bin_size)
    v_tgt = population_rate_vector(spike_target, pop_indices, n_steps, bin_size)

    norm_sim = np.linalg.norm(v_sim)
    norm_tgt = np.linalg.norm(v_tgt)
    if norm_sim < 1e-12 or norm_tgt < 1e-12:
        return 1.0  # degenerate — return max dissimilarity
    cosine = np.dot(v_sim, v_tgt) / (norm_sim * norm_tgt)
    return float(1.0 - cosine)


# ── 2. Fleiss Kappa (burst-timing synchrony) ───────────────────────────────────

def _spike_binary_matrix(
    spikes:     Dict[int, np.ndarray],
    all_cells:  List[int],
    n_steps:    int,
    bin_size:   int,
) -> np.ndarray:
    """Binary spike matrix: shape (n_cells, n_bins)."""
    n_bins = n_steps // bin_size
    mat = np.zeros((len(all_cells), n_bins), dtype=np.int32)
    for row, idx in enumerate(all_cells):
        if idx in spikes:
            bins = spikes[idx] // bin_size
            valid = bins[bins < n_bins]
            mat[row, valid] = 1
    return mat


def compute_fleiss_kappa(
    spikes:    Dict[int, np.ndarray],
    layers:    Dict[str, List[int]],   # {"L23": [...], "L4": [...], "L56": [...]}
    n_steps:   int,
    bin_size:  int = 200,
) -> float:
    """
    Fleiss' Kappa measuring burst-timing agreement across laminar layers.

    Returns:
        kappa ∈ (-1, 1). Target: kappa < 0.10 (physiological asynchrony).
    """
    all_cells = []
    for cells in layers.values():
        all_cells.extend(cells)
    if not all_cells:
        return 0.0

    mat = _spike_binary_matrix(spikes, all_cells, n_steps, bin_size)
    n_bins  = mat.shape[1]
    n_cells = mat.shape[0]

    if n_cells < 2 or n_bins < 2:
        return 0.0

    # Proportion of raters (cells) assigning each rating (spike / no-spike)
    # n_ij: matrix (n_bins × 2), count of cells spiking/not-spiking per bin
    n_spike    = mat.sum(axis=0)            # (n_bins,)
    n_nospike  = n_cells - n_spike          # (n_bins,)
    n_ij       = np.stack([n_spike, n_nospike], axis=1).astype(float)  # (n_bins, 2)

    # Per-subject agreement
    P_i = (np.sum(n_ij * (n_ij - 1), axis=1) /
           max(n_cells * (n_cells - 1), 1e-12))

    P_bar = float(np.mean(P_i))

    # Marginal proportions
    p_j = n_ij.sum(axis=0) / max(n_ij.sum(), 1e-12)
    P_e = float(np.sum(p_j ** 2))

    if abs(1.0 - P_e) < 1e-12:
        return 0.0

    kappa = (P_bar - P_e) / (1.0 - P_e)
    return float(np.clip(kappa, -1.0, 1.0))


# ── 3. Spectral Similarity Score ───────────────────────────────────────────────

CANONICAL_BANDS = {
    "theta": ( 4,  8),
    "alpha": ( 8, 13),
    "beta":  (13, 30),
    "gamma": (30, 80),
}


def compute_band_power(
    lfp: np.ndarray,
    fs:  float,
    bands: Optional[Dict[str, Tuple[float, float]]] = None,
) -> Dict[str, float]:
    """Welch PSD, then integrate per frequency band. Returns normalised powers."""
    if bands is None:
        bands = CANONICAL_BANDS
    freqs, psd = welch(lfp, fs=fs, nperseg=min(256, len(lfp) // 4))
    band_powers = {}
    total_power = np.trapz(psd, freqs) + 1e-30
    for name, (f_lo, f_hi) in bands.items():
        mask = (freqs >= f_lo) & (freqs <= f_hi)
        band_powers[name] = float(np.trapz(psd[mask], freqs[mask]) / total_power)
    return band_powers


def compute_sss(
    lfp_sim:    np.ndarray,
    lfp_target: np.ndarray,
    fs:         float,
    bands:      Optional[Dict[str, Tuple[float, float]]] = None,
) -> float:
    """
    Spectral Similarity Score: mean squared relative error per band.
    Range [0, ∞).  0 = perfect spectral match.

    Empirical targets (Bastos omission context, alpha/beta dominant):
        alpha ≈ 0.35, beta ≈ 0.40, gamma ≈ 0.05
    """
    if bands is None:
        bands = CANONICAL_BANDS
    bp_sim = compute_band_power(lfp_sim, fs, bands)
    bp_tgt = compute_band_power(lfp_target, fs, bands)
    errors = []
    for band in bands:
        s = bp_sim.get(band, 0.0)
        t = bp_tgt.get(band, 0.0) + 1e-10
        errors.append(((s - t) / t) ** 2)
    return float(np.mean(errors))


def empirical_omission_target_lfp(
    t_ms: np.ndarray,
    fs: float,
) -> np.ndarray:
    """
    Synthetic empirical target: alpha/beta-dominant LFP for omission context.
    Represents the Bastos et al. observation of global inhibition via alpha/beta.

    Returns a 1D LFP-like signal (arbitrary units) with:
      - Strong alpha (10 Hz) + beta (20 Hz) components
      - Weak gamma (40 Hz)
    """
    t_s = t_ms / 1000.0
    target = (
        1.0  * np.sin(2 * np.pi * 10.0 * t_s) +   # alpha
        0.8  * np.sin(2 * np.pi * 20.0 * t_s) +   # beta
        0.1  * np.sin(2 * np.pi * 40.0 * t_s) +   # weak gamma
        0.05 * np.random.randn(len(t_s))           # background noise
    )
    return target
