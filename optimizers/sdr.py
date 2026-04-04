import jax
import jax.numpy as jnp
import jaxley as jx
import optax
import numpy as np
from typing import Callable, Any
from flax.struct import dataclass

# ClampTransform removed (use jnp.clip)

@dataclass
class SDRState:
    momentum_accum: Any
    step_count: int

def SDR(
    learning_rate: float = 1e-2,
    momentum: float = 0.9,
    sigma: float = 0.1,
    change_lower_bound: float = -1.0,
    change_upper_bound: float = 1.0,
    delta_distribution: Callable = jax.random.normal
) -> optax.GradientTransformation:
    """
    Stochastic Delta Rule (SDR) optimizer.
    """
    def init_fn(params):
        momentum_accum = jax.tree.map(jnp.zeros_like, params)
        return SDRState(momentum_accum=momentum_accum, step_count=0)

    def update_fn(updates, state, params=None, key=None):
        if key is None:
            raise ValueError("SDR requires a random 'key' to be passed to update().")
        
        grads = updates
        new_momentum_accum = jax.tree.map(
            lambda m, g: momentum * m + g,
            state.momentum_accum, grads
        )

        grad_signs = jax.tree.map(jnp.sign, new_momentum_accum)

        param_leaves, treedef = jax.tree.flatten(grads)
        subkeys = jax.random.split(key, len(param_leaves))
        param_keys_tree = jax.tree.unflatten(treedef, subkeys)

        random_factors = jax.tree.map(
            lambda g, k: sigma * delta_distribution(k, g.shape),
            grads, param_keys_tree
        )

        from jbiophysics.utils.math import apply_spatial_smoothing

        random_factors = jax.tree.map(apply_spatial_smoothing, random_factors)

        raw_updates = jax.tree.map(
            lambda s, r: -learning_rate * s * r,
            grad_signs, random_factors
        )

        final_updates = jax.tree.map(lambda x: jnp.clip(x, change_lower_bound, change_upper_bound), raw_updates)

        new_state = SDRState(
            momentum_accum=new_momentum_accum,
            step_count=state.step_count + 1
        )
        return final_updates, new_state

    return optax.GradientTransformation(init_fn, update_fn)

# --- GSDR (Genetic Stochastic Delta Rule) ---