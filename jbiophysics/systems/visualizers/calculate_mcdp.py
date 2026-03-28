import jax
import jax.numpy as jnp
import jaxley as jx
import optax

def calculate_mcdp(traces, ampa_pre_inds, ampa_post_inds, gaba_pre_inds, gaba_post_inds):
    """
    Calculates normalized Mutual-correlation dependent plasticity (MCDP) factors 
    for all trainable parameters.
    """
    r_ampa = compute_correlations(traces, ampa_pre_inds, ampa_post_inds)
    r_gaba = compute_correlations(traces, gaba_pre_inds, gaba_post_inds)

    r_ampa_norm = (r_ampa - jnp.mean(r_ampa)) / (jnp.std(r_ampa) + 1e-6) if r_ampa.size > 0 else jnp.array([])
    r_gaba_norm = (r_gaba - jnp.mean(r_gaba)) / (jnp.std(r_gaba) + 1e-6) if r_gaba.size > 0 else jnp.array([])

    return [{'gAMPA': r_ampa_norm}, {'gGABAa': r_gaba_norm}]