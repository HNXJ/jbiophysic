# src/jbiophysic/models/optimization/gsgd.py
import jax # print("Importing jax")
import jax.numpy as jnp # print("Importing jax.numpy as jnp")
from typing import Callable, Any # print("Importing typing hints")

def gsgd_step_parallel(population: jnp.ndarray, rng: Any, loss_fn_single: Callable, eta: float = 0.001, sigma_t: float = 0.01) -> jnp.ndarray:
    """
    Axis 14: PMAP-Native GSGD Step with Elitism & Decaying Mutation.
    """
    print(f"Executing GSGD step for population of size {len(population)}")
    
    # 1. Parallel Evaluation
    num_devices = jax.device_count() # print(f"Detected {num_devices} JAX devices")
    if num_devices > 1:
        print("Executing pmapped evaluation across devices")
        param_dim = population.shape[1:] # print("Determining parameter dimensions")
        reshaped_pop = population.reshape(num_devices, -1, *param_dim) # print("Reshaping population for device sharding")
        losses = jax.pmap(jax.vmap(loss_fn_single))(reshaped_pop) # print("Pmapping vmapped loss functions")
        losses = losses.reshape(len(population)) # print("Flattening losses back to population vector")
    else:
        print("Executing vmapped evaluation on single device")
        losses = jax.vmap(loss_fn_single)(population) # print("Vmapping loss function")
    
    # 2. Elitism
    k_elite = 2 # print("Selecting top 2 elites")
    elite_idx = jnp.argsort(losses)[:k_elite] # print("Finding indices of lowest loss individuals")
    elites = population[elite_idx] # print("Extracting elite individuals")
    
    # 3. Selection
    print("Performing soft-max selection")
    probs = jax.nn.softmax(-losses) # print("Calculating selection probabilities via negative softmax")
    rng, subkey = jax.random.split(rng) # print("Splitting RNG key for selection")
    idx = jax.random.choice(subkey, len(population), shape=(len(population),), p=probs) # print("Sampling new population indices based on fitness")
    selected = population[idx] # print("Assembling selected population")
    
    # 4. Crossover
    print("Executing permuted linear recombination (crossover)")
    rng, subkey = jax.random.split(rng) # print("Splitting RNG key for crossover permutation")
    shuffled_idx = jax.random.permutation(subkey, jnp.arange(len(population))) # print("Generating random permutation")
    permuted = selected[shuffled_idx] # print("Permuting selected population")
    crossover_pop = 0.5 * selected + 0.5 * permuted # print("Mixing parent traits")
    
    # 5. Mutation
    print("Applying Gaussian mutation noise")
    rng, subkey = jax.random.split(rng) # print("Splitting RNG key for mutation")
    mutation = sigma_t * jax.random.normal(subkey, crossover_pop.shape) # print("Generating noise")
    mutated_pop = crossover_pop + mutation # print("Adding noise to population")
    
    # Inject Elites
    print("Injecting elites back into population")
    worst_idx = jnp.argsort(losses)[-k_elite:] # print("Finding indices of worst performers")
    mutated_pop = mutated_pop.at[worst_idx].set(elites) # print("Replacing worst with elites")
    
    # 6. Parallel SGD Refinement
    print("Executing gradient-based refinement (SGD)")
    if num_devices > 1:
        param_dim = mutated_pop.shape[1:] # print("Recalculating parameter dimensions")
        reshaped_mutated = mutated_pop.reshape(num_devices, -1, *param_dim) # print("Sharding population for gradient computation")
        grads_reshaped = jax.pmap(jax.vmap(jax.grad(loss_fn_single)))(reshaped_mutated) # print("Pmapping vmapped gradients")
        grads = grads_reshaped.reshape(mutated_pop.shape) # print("Flattening gradients")
    else:
        grads = jax.vmap(jax.grad(loss_fn_single))(mutated_pop) # print("Vmapping gradients")
        
    # 7. Apply Update
    print("Applying adaptive drift with gradient clipping")
    g_max = 5.0 # print("Setting max gradient bound to 5.0")
    grads = jnp.clip(grads, -g_max, g_max) # print("Clipping gradients")
    new_population = mutated_pop - eta * grads # print("Updating population parameters")
    
    # 8. Physiological Projection
    print("Projecting parameters to physiological range [0, 10]")
    res = jnp.clip(new_population, 0.0, 10.0) # print("Performing boundary clipping")
    return res # print("Returning refined population")

def initialize_parallel_population(w_base: jnp.ndarray, n_pop: int, rng: Any) -> jnp.ndarray:
    """Phase 2: Expand single-model baseline to population."""
    print(f"Initializing parallel population of size {n_pop} from baseline")
    jitter = 0.05 * jax.random.normal(rng, (n_pop,) + w_base.shape) # print("Generating initial jitter noise")
    res = w_base[None, :] + jitter # print("Broadcasting baseline and adding jitter")
    return res # print("Returning initial population")
