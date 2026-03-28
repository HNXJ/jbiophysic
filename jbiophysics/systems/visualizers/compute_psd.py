import jax
import jax.numpy as jnp
import jaxley as jx
import optax

def compute_psd(signal_1d, dt, target_freqs=None):
    """Computes PSD for a 1D signal."""
    N = signal_1d.shape[-1]
    fs = 1000.0 / dt
    signal_fft = jnp.fft.rfft(signal_1d)
    freqs = jnp.fft.rfftfreq(N, d=dt/1000.0)
    psd_raw = jnp.abs(signal_fft)**2 / (N * fs)
    
    if target_freqs is None:
        target_freqs = jnp.linspace(1.0, 100.0, 100)
    
    interpolated_psd = jnp.interp(target_freqs, freqs, psd_raw)
    return target_freqs, interpolated_psd