import jax
import jax.numpy as jnp
import jaxley as jx
import optax
import numpy as np
from typing import Callable, Any
from flax.struct import dataclass

class ClampTransform:
    def __init__(self, lower, upper):
        self.lower = lower
        self.upper = upper
    def forward(self, x):
        return jnp.clip(x, self.lower, self.upper)

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

        def smooth_factor(x):
            if x.ndim == 2:
                n, m = x.shape
                kn = max(1, int(np.sqrt(n)))
                km = max(1, int(np.sqrt(m)))
                kernel = jnp.ones((kn, km)) / (kn * km)
                return jax.scipy.signal.convolve2d(x, kernel, mode='same')
            elif x.ndim == 1:
                n = x.shape[0]
                k = max(1, int(np.sqrt(n)))
                kernel = jnp.ones((k,)) / k
                return jnp.convolve(x, kernel, mode='same')
            return x

        random_factors = jax.tree.map(smooth_factor, random_factors)

        raw_updates = jax.tree.map(
            lambda s, r: -learning_rate * s * r,
            grad_signs, random_factors
        )

        boundTransform = ClampTransform(change_lower_bound, change_upper_bound)
        final_updates = jax.tree.map(lambda x: boundTransform.forward(x), raw_updates)

        new_state = SDRState(
            momentum_accum=new_momentum_accum,
            step_count=state.step_count + 1
        )
        return final_updates, new_state

    return optax.GradientTransformation(init_fn, update_fn)

# --- GSDR (Genetic Stochastic Delta Rule) ---