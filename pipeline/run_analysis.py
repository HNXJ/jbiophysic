# pipeline/run_analysis.py
import jax.numpy as jnp
import numpy as np
import scipy.signal
import hashlib
import json
import os

def generate_cache_key(data_array, params):
    """Axis 16: Lightweight SHA256 Caching to avoid redundant processing."""
    # Fast shape + stats hashing rather than raw byte hashing
    stats = f"mean:{np.mean(data_array):.5f}_var:{np.var(data_array):.5f}"
    param_bytes = json.dumps(params, sort_keys=True).encode('utf-8')
    return hashlib.sha256((str(np.asarray(data_array).shape) + stats).encode('utf-8') + param_bytes).hexdigest()

def compute_spectral_features(lfp_signals, fs=1000.0, apply_window=True, use_cache=True):
    """
    Axis 14: High-performance LFP pipeline with windowing and JAX FFT.
    Produces Time-Frequency Representations (TFR) and bandpowers.
    """
    params = {"fs": fs, "window": apply_window}
    cache_key = generate_cache_key(lfp_signals, params)
    cache_path = f"pipeline/.cache/{cache_key}.json"
    
    if use_cache and os.path.exists(cache_path):
        print("⚡ Loading LFP analysis from cache...")
        with open(cache_path, "r") as f:
            return json.load(f)
            
    # Number of time steps
    n_steps = lfp_signals.shape[-1]
    
    # Axis 16: Hann Windowing broadcasted
    if apply_window:
        window = scipy.signal.windows.hann(n_steps)
        if len(lfp_signals.shape) > 1:
            window = window[None, :]
        lfp_signals = lfp_signals * window

    # Axis 16: True PSD Biological normalization
    freqs = jnp.fft.rfftfreq(n_steps, d=1/fs)
    fft_mag = jnp.abs(jnp.fft.rfft(jnp.array(lfp_signals), axis=-1))
    psd = (fft_mag ** 2) / n_steps
    
    # Power bands
    gamma_pwr = jnp.mean(psd[..., (freqs >= 30) & (freqs <= 80)], axis=-1)
    beta_pwr = jnp.mean(psd[..., (freqs >= 13) & (freqs <= 30)], axis=-1)
    
    results = {
        "gamma_power": np.array(gamma_pwr).tolist(),
        "beta_power": np.array(beta_pwr).tolist(),
        "peak_freq": float(freqs[jnp.argmax(fft_mag, axis=-1)[0]]),
    }
    
    if use_cache:
        os.makedirs("pipeline/.cache", exist_ok=True)
        with open(cache_path, "w") as f:
            json.dump(results, f)
            
    return results

if __name__ == "__main__":
    # Test pipeline
    mock_lfp = np.random.normal(0, 1, (10, 5000))
    res = compute_spectral_features(mock_lfp)
    print("✅ Analysis Pipeline Refined (Hann Windowing + JAX FFT + Caching).")
