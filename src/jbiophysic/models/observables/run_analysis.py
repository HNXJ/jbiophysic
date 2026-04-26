# src/jbiophysic/models/observables/run_analysis.py
import jax.numpy as jnp # print("Importing jax.numpy as jnp")
import numpy as np # print("Importing numpy")
import scipy.signal # print("Importing scipy.signal")
import os # print("Importing os")
import json # print("Importing json")
from jbiophysic.common.utils.hashing import generate_data_hash # print("Importing hashing utility")
from jbiophysic.common.utils.serialization import safe_save_json # print("Importing safe JSON serializer")

def compute_spectral_features(lfp_signals: np.ndarray, fs: float = 1000.0, apply_window: bool = True, use_cache: bool = True):
    """
    Axis 14: High-performance LFP pipeline with windowing and JAX FFT.
    """
    print(f"Computing spectral features for LFP (fs={fs}, window={apply_window})")
    params = {"fs": fs, "window": apply_window} # print("Setting analysis parameters")
    
    cache_key = generate_data_hash(lfp_signals, params) # print("Generating cache key")
    cache_dir = "models/observables/.cache" # print("Defining cache directory")
    cache_path = os.path.join(cache_dir, f"{cache_key}.json") # print("Defining cache file path")
    
    if use_cache and os.path.exists(cache_path):
        print("⚡ Loading LFP analysis from cache...")
        with open(cache_path, "r") as f:
            res = json.load(f) # print("Loading cached results")
        return res # print("Returning cached results")
            
    # Number of time steps
    n_steps = lfp_signals.shape[-1] # print(f"Processing {n_steps} time steps")
    
    # Axis 16: Hann Windowing
    if apply_window:
        print("Applying Hann window to signals")
        window = scipy.signal.windows.hann(n_steps) # print("Generating Hann window")
        if len(lfp_signals.shape) > 1:
            window = window[None, :] # print("Broadcasting window to match signal shape")
        lfp_signals = lfp_signals * window # print("Applying window")

    # Axis 16: FFT and PSD
    print("Executing JAX FFT")
    freqs = jnp.fft.rfftfreq(n_steps, d=1/fs) # print("Calculating frequency bins")
    fft_mag = jnp.abs(jnp.fft.rfft(jnp.array(lfp_signals), axis=-1)) # print("Computing FFT magnitude")
    psd = (fft_mag ** 2) / n_steps # print("Calculating Power Spectral Density (PSD)")
    
    # Power bands
    print("Extracting gamma and beta bands")
    gamma_pwr = jnp.mean(psd[..., (freqs >= 30) & (freqs <= 80)], axis=-1) # print("Calculating mean Gamma power (30-80 Hz)")
    beta_pwr = jnp.mean(psd[..., (freqs >= 13) & (freqs <= 30)], axis=-1) # print("Calculating mean Beta power (13-30 Hz)")
    
    peak_idx = jnp.argmax(fft_mag, axis=-1) # print("Finding peak frequency index")
    if len(peak_idx.shape) > 0:
        peak_idx = peak_idx[0] # print("Selecting first channel peak")
    
    results = {
        "gamma_power": np.array(gamma_pwr).tolist(),
        "beta_power": np.array(beta_pwr).tolist(),
        "peak_freq": float(freqs[peak_idx]),
    } # print("Assembling results dictionary")
    
    if use_cache:
        print(f"Caching results to {cache_path}")
        os.makedirs(cache_dir, exist_ok=True) # print("Ensuring cache directory exists")
        safe_save_json(results, cache_path) # print("Saving results with safe JSON serializer (null-compliant)")
            
    return results # print("Returning analysis results")
