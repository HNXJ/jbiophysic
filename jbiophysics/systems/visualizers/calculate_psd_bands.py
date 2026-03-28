import jax
import jax.numpy as jnp
import jaxley as jx
import optax

def calculate_psd_bands(psd, freqs, band_defs):
    """Calculates average power in specified frequency bands."""
    band_powers = {}
    for band_name, (f_min, f_max) in band_defs.items():
        indices = (freqs >= f_min) & (freqs < f_max)
        if not jnp.any(indices):
            band_powers[band_name] = 0.0
        else:
            band_powers[band_name] = jnp.mean(psd[..., indices])
    return band_powers