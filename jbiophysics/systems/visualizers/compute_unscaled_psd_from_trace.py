import jax
import jax.numpy as jnp
import jaxley as jx
import optax

def compute_unscaled_psd_from_trace(trace, dt_global):
    """Computes unscaled PSD for a 1D trace."""
    N = trace.shape[-1]
    fs = 1000.0 / dt_global
    signal_fft = jnp.fft.rfft(trace)
    freqs = jnp.fft.rfftfreq(N, d=dt_global/1000.0)
    psd_raw = jnp.abs(signal_fft)**2 / (N * fs)
    target_freqs = jnp.linspace(1.0, 100.0, 100)
    interpolated_psd = jnp.interp(target_freqs, freqs, psd_raw)
    return target_freqs, interpolated_psd