import jax
import jax.numpy as jnp
import numpy as np

def compute_rsa(spikes_sim, spikes_tgt, pop_indices, n_steps):
    """Correlation between population firing rate vectors (RSA)."""
    def get_fr(spikes, idx):
        # spikes: {cell_id: [spike_times]}
        v = jnp.zeros((n_steps,))
        for i in idx:
            if i in spikes:
                v = v.at[jnp.array(spikes[i])].add(1.0)
        return v

    fr_sim = get_fr(spikes_sim, pop_indices)
    fr_tgt = get_fr(spikes_tgt, pop_indices)
    
    # Cosine Similarity (↓ is 1-cosim)
    num = jnp.dot(fr_sim, fr_tgt)
    den = jnp.linalg.norm(fr_sim) * jnp.linalg.norm(fr_tgt) + 1e-12
    return 1.0 - (num / den)

def compute_fleiss_kappa(spikes, layers, n_steps):
    """Synchrony metric across populations (Fleiss Kappa)."""
    # Simply calculating variance of population average firing rate (burstiness)
    # We want Kappa < 0.1 for asynchronous states
    all_indices = []
    for l in layers.values(): all_indices.extend(l)
    
    pop_total = jnp.zeros((n_steps,))
    for i in all_indices:
        if i in spikes: pop_total = pop_total.at[jnp.array(spikes[i])].add(1.0)
    
    return jnp.var(pop_total) / (jnp.mean(pop_total) + 1e-12)

def compute_sss(lfp_sim, lfp_tgt, fs):
    """Spectral Similarity Score (SSS) - 1.0 - corr(log(PSD))."""
    from viz.psd import compute_psd
    _, psd_sim = compute_psd(lfp_sim, 1000.0/fs)
    _, psd_tgt = compute_psd(lfp_tgt, 1000.0/fs)
    
    l_sim = jnp.log(psd_sim + 1e-6)
    l_tgt = jnp.log(psd_tgt + 1e-6)
    
    corr = jnp.corrcoef(l_sim, l_tgt)[0, 1]
    return 1.0 - corr

def empirical_omission_target_lfp(n_steps):
    # Dummy alpha-beta target (e.g., 20Hz beta, 10Hz alpha mixture)
    t = jnp.arange(n_steps) * 0.001
    return 0.5 * jnp.sin(2 * jnp.pi * 10 * t) + 0.3 * jnp.sin(2 * jnp.pi * 20 * t)
