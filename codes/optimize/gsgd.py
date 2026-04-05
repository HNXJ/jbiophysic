# codes/optimize/gsgd.py
import jax
import jax.numpy as jnp
from typing import Callable, Any

def gsgd_step_parallel(population, rng, loss_fn_single: Callable, eta=0.001, sigma_t=0.01):
    """
    Axis 14: PMAP-Native GSGD Step with Elitism & Decaying Mutation.
    Selection -> Crossover -> Mutation -> Parallel SGD Refinement.
    """
    # 1. Parallel Evaluation
    # Note: Requires a machine with multiple devices to fully utilize.
    # Fallback to vmap internally if only 1 device is present, but pmap is the target.
    num_devices = jax.device_count()
    if num_devices > 1:
        # Assuming population size is divisible by num_devices
        losses = jax.pmap(loss_fn_single)(population)
    else:
        losses = jax.vmap(loss_fn_single)(population)
    
    # 2. Elitism (Tracking the Top-K)
    k_elite = 2
    elite_idx = jnp.argsort(losses)[:k_elite]
    elites = population[elite_idx]
    
    # 3. Selection (Softmax over population fitness)
    probs = jax.nn.softmax(-losses)
    rng, subkey = jax.random.split(rng)
    idx = jax.random.choice(subkey, len(population), shape=(len(population),), p=probs)
    selected = population[idx]
    
    # 4. Crossover (Permuted linear recombination)
    rng, subkey = jax.random.split(rng)
    shuffled_idx = jax.random.permutation(subkey, jnp.arange(len(population)))
    permuted = selected[shuffled_idx]
    crossover_pop = 0.5 * selected + 0.5 * permuted
    
    # 5. Mutation (Decaying Gaussian noise)
    rng, subkey = jax.random.split(rng)
    mutation = sigma_t * jax.random.normal(subkey, crossover_pop.shape)
    mutated_pop = crossover_pop + mutation
    
    # Inject Elites to prevent regression
    mutated_pop = mutated_pop.at[:k_elite].set(elites)
    
    # 6. Parallel SGD Refinement (Pmapped jax.grad)
    if num_devices > 1:
        grads = jax.pmap(jax.grad(loss_fn_single))(mutated_pop)
    else:
        grads = jax.vmap(jax.grad(loss_fn_single))(mutated_pop)
        
    # 7. Apply Refined Update (AGSDR Adaptive Drift)
    g_max = 5.0 # Max grad bound for clipping
    grads = jnp.clip(grads, -g_max, g_max)
    new_population = mutated_pop - eta * grads
    
    # 8. Physiological Projection (Clipping)
    return jnp.clip(new_population, 0.0, 10.0)

def initialize_parallel_population(w_base, n_pop, rng):
    """Phase 2: Expand single-model baseline to population."""
    jitter = 0.05 * jax.random.normal(rng, (n_pop,) + w_base.shape)
    return w_base[None, :] + jitter
