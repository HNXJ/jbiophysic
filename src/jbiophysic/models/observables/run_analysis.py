# src/jbiophysic/models/observables/run_analysis.py
from jbiophysic.common.utils.logging import get_logger

logger = get_logger(__name__)

import jax.numpy as jnp
import numpy as np
import scipy.signal
import os
import json
from jbiophysic.common.utils.hashing import generate_data_hash
from jbiophysic.common.utils.serialization import safe_save_json

def compute_spectral_features(lfp_signals: np.ndarray, fs: float = 1000.0, apply_window: bool = True, use_cache: bool = True):
    """
    Axis 14: High-performance LFP pipeline with windowing and JAX FFT.
    """
    logger.info(f"Computing spectral features for LFP (fs={fs}, window={apply_window})")
    params = {"fs": fs, "window": apply_window}
    
    cache_key = generate_data_hash(lfp_signals, params)
    cache_dir = "models/observables/.cache"
    cache_path = os.path.join(cache_dir, f"{cache_key}.json")
    
    if use_cache and os.path.exists(cache_path):
        logger.info("⚡ Loading LFP analysis from cache...")
        with open(cache_path, "r") as f:
            res = json.load(f)
        return res
            
    # Number of time steps
    n_steps = lfp_signals.shape[-1]
    
    # Axis 16: Hann Windowing
    if apply_window:
        logger.info("Applying Hann window to signals")
        window = scipy.signal.windows.hann(n_steps)
        if len(lfp_signals.shape) > 1:
            window = window[None, :]
        lfp_signals = lfp_signals * window

    # Axis 16: FFT and PSD
    logger.info("Executing JAX FFT")
    freqs = jnp.fft.rfftfreq(n_steps, d=1/fs)
    fft_mag = jnp.abs(jnp.fft.rfft(jnp.array(lfp_signals), axis=-1))
    psd = (fft_mag ** 2) / n_steps
    
    # Power bands
    logger.info("Extracting gamma and beta bands")
    gamma_pwr = jnp.mean(psd[..., (freqs >= 30) & (freqs <= 80)], axis=-1)
    beta_pwr = jnp.mean(psd[..., (freqs >= 13) & (freqs <= 30)], axis=-1)
    
    peak_idx = jnp.argmax(fft_mag, axis=-1)
    if len(peak_idx.shape) > 0:
        peak_idx = peak_idx[0]
    
    results = {
        "gamma_power": np.array(gamma_pwr).tolist(),
        "beta_power": np.array(beta_pwr).tolist(),
        "peak_freq": float(freqs[peak_idx]),
    }
    
    if use_cache:
        logger.info(f"Caching results to {cache_path}")
        os.makedirs(cache_dir, exist_ok=True)
        safe_save_json(results, cache_path)
            
    return results
