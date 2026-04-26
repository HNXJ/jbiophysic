# src/jbiophysic/models/optimization/gsgd.py
from jbiophysic.common.utils.logging import get_logger

logger = get_logger(__name__)

import jax
import jax.numpy as jnp
from typing import Callable, Any

def gsgd_step_parallel(population: jnp.ndarray, rng: Any, loss_fn_single: Callable, eta: float = 0.001, sigma_t: float = 0.01) -> jnp.ndarray:
    """
    Axis 14: PMAP-Native GSGD Step with Elitism & Decaying Mutation.
    """
def gsgd_step_parallel(population: jnp.ndarray, rng: Any, loss_fn_single: Callable, eta: float = 0.001, sigma_t: float = 0.01) -> jnp.ndarray:
    """
    Axis 14: PMAP-Native GSGD Step with Elitism & Decaying Mutation.
    """
    # 1. Parallel Evaluation
    num_devices = jax.device_count()
    if num_devices > 1:
        param_dim = population.shape[1:]
        reshaped_pop = population.reshape(num_devices, -1, *param_dim)
        losses = jax.pmap(jax.vmap(loss_fn_single))(reshaped_pop)
        losses = losses.reshape(len(population))
    else:
        losses = jax.vmap(loss_fn_single)(population)
    
    # 2. Elitism
    k_elite = 2
    elite_idx = jnp.argsort(losses)[:k_elite]
    elites = population[elite_idx]
    
    # 3. Selection
    probs = jax.nn.softmax(-losses)
    rng, subkey = jax.random.split(rng)
    idx = jax.random.choice(subkey, len(population), shape=(len(population),), p=probs)
    selected = population[idx]
    
    # 4. Crossover
    rng, subkey = jax.random.split(rng)
    shuffled_idx = jax.random.permutation(subkey, jnp.arange(len(population)))
    permuted = selected[shuffled_idx]
    crossover_pop = 0.5 * selected + 0.5 * permuted
    
    # 5. Mutation
    rng, subkey = jax.random.split(rng)
    mutation = sigma_t * jax.random.normal(subkey, crossover_pop.shape)
    mutated_pop = crossover_pop + mutation
    
    # Inject Elites
    worst_idx = jnp.argsort(losses)[-k_elite:]
    mutated_pop = mutated_pop.at[worst_idx].set(elites)
    
    # 6. Parallel SGD Refinement
    if num_devices > 1:
        param_dim = mutated_pop.shape[1:]
        reshaped_mutated = mutated_pop.reshape(num_devices, -1, *param_dim)
        grads_reshaped = jax.pmap(jax.vmap(jax.grad(loss_fn_single)))(reshaped_mutated)
        grads = grads_reshaped.reshape(mutated_pop.shape)
    else:
        grads = jax.vmap(jax.grad(loss_fn_single))(mutated_pop)
        
    # 7. Apply Update
    g_max = 5.0
    grads = jnp.clip(grads, -g_max, g_max)
    new_population = mutated_pop - eta * grads
    
    # 8. Physiological Projection
    res = jnp.clip(new_population, 0.0, 10.0)
    return res

def initialize_parallel_population(w_base: jnp.ndarray, n_pop: int, rng: Any) -> jnp.ndarray:
    """Phase 2: Expand single-model baseline to population."""
    jitter = 0.05 * jax.random.normal(rng, (n_pop,) + w_base.shape)
    res = w_base[None, :] + jitter
    return res
