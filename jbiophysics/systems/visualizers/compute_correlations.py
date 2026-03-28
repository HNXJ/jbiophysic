import jax
import jax.numpy as jnp
import jaxley as jx
import optax

def compute_correlations(traces, pre_inds, post_inds):
    """
    Computes mutual Pearson correlation (est., time adjusted) between pre and post traces.
    traces: (Batch, Cells, Time)
    pre_inds, post_inds: (Num_Edges,)
    Returns: (Num_Edges,)
    """
    if pre_inds.size == 0 or post_inds.size == 0:
        return jnp.array([])

    # Shifted correlation
    pre_v = traces[:, pre_inds, :-10]
    post_v = traces[:, post_inds, 10:]

    pre_mean = jnp.mean(pre_v, axis=-1, keepdims=True)
    post_mean = jnp.mean(post_v, axis=-1, keepdims=True)
    
    num = jnp.sum((pre_v - pre_mean) * (post_v - post_mean), axis=-1)
    den = jnp.sqrt(jnp.sum((pre_v - pre_mean)**2, axis=-1) * jnp.sum((post_v - post_mean)**2, axis=-1))
    
    # Square correlation to get mutual information proxy
    return jnp.mean(num**2 / (den + 1e-6), axis=0)