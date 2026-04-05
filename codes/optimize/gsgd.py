# codes/optimize/gsgd.py
import jax
import jax.numpy as jnp
from typing import Callable, Any

def gsgd_step_parallel(population, rng, loss_fn_vmap: Callable, eta=0.001):
    """
    Axis 12: Parallel-Native GSGD Step.
    Selection -> Crossover -> Mutation -> Parallel SGD Refinement.
    """
    # 1. Parallel Evaluation
    losses = loss_fn_vmap(population)
    
    # 2. Selection (Softmax over population)
    probs = jax.nn.softmax(-losses)
    rng, subkey = jax.random.split(rng)
    idx = jax.random.choice(subkey, len(population), shape=(len(population),), p=probs)
    selected = population[idx]
    
    # 3. Crossover (Permuted linear recombination)
    rng, subkey = jax.random.split(rng)
    shuffled_idx = jax.random.permutation(subkey, jnp.arange(len(population)))
    permuted = selected[shuffled_idx]
    crossover_pop = 0.5 * selected + 0.5 * permuted
    
    # 4. Mutation (Gaussian noise)
    rng, subkey = jax.random.split(rng)
    mutation = 0.01 * jax.random.normal(subkey, crossover_pop.shape)
    mutated_pop = crossover_pop + mutation
    
    # 5. Parallel SGD Refinement (Vmapped jax.grad)
    # We vmap over the gradient of the loss function
    def grad_fn(w): return jax.grad(loss_fn_vmap)(w) # This is a placeholder for actual vmapped grad
    
    # In practice, for JAX efficiency, we use vmap(grad(loss_fn))
    grads = jax.vmap(jax.grad(loss_fn_vmap.func))(mutated_pop) if hasattr(loss_fn_vmap, 'func') else jax.vmap(jax.grad(loss_fn_vmap))(mutated_pop)
    
    # 6. Apply Refined Update (AGSDR Adaptive Drift)
    grads = jnp.clip(grads, -1.0, 1.0)
    new_population = mutated_pop - eta * grads
    
    # 7. Physiological Projection (Clipping)
    return jnp.clip(new_population, 0.0, 10.0)

def initialize_parallel_population(w_base, n_pop, rng):
    """Phase 2: Expand single-model baseline to population."""
    jitter = 0.05 * jax.random.normal(rng, (n_pop,) + w_base.shape)
    return w_base[None, :] + jitter
