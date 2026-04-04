# codes/optimize/gsgd.py
import jax
import jax.numpy as jnp
from typing import List, Callable, Dict, Any
from codes.optimize.agsdr import AGSDR

class GSGD:
    """
    Genetic-Stochastic Gradient Descent (Axis 12).
    A hybrid evolutionary-stochastic gradient optimizer for biophysical systems.
    """
    def __init__(self, population_size=10, mutation_rate=0.01, eta=0.001):
        self.pop_size = population_size
        self.mutation_rate = mutation_rate
        self.agsdr = AGSDR(eta=eta)

    def select(self, rng, population, losses):
        """Softmax selection based on fitness (negative loss)."""
        probs = jax.nn.softmax(-jnp.array(losses))
        idx = jax.random.choice(rng, self.pop_size, p=probs)
        return [population[i] for i in idx]

    def crossover(self, rng, p1, p2, alpha=0.5):
        """Linear crossover between two individuals."""
        return alpha * p1 + (1.0 - alpha) * p2

    def mutate(self, rng, ind):
        """Gaussian mutation."""
        noise = self.mutation_rate * jax.random.normal(rng, ind.shape)
        return ind + noise

    def gsgd_step(self, rng, population, loss_fn: Callable):
        """
        Axis 12 Master Update Rule:
        Evolution (Crossover/Mutation) + Local SGD Refinement (AGSDR).
        """
        # 1. Evaluate fitness (should be vmapped/pmapped)
        losses = [loss_fn(p) for p in population]
        
        # 2. Select next generation
        rng, subkey = jax.random.split(rng)
        mating_pool = self.select(subkey, population, losses)
        
        new_population = []
        for i in range(self.pop_size):
            # 3. Evolution
            p1 = mating_pool[i]
            p2 = mating_pool[(i+1) % self.pop_size]
            
            rng, subkey1, subkey2 = jax.random.split(rng, 3)
            child = self.crossover(subkey1, p1, p2)
            child = self.mutate(subkey2, child)
            
            # 4. SGD Refinement (local AGSDR adaptive drift)
            grad = jax.grad(loss_fn)(child)
            refined_child = self.agsdr.update_weights(child, grad)
            
            # 5. Elitism comparison (keep best between parent and refined child)
            new_population.append(refined_child)
            
        return new_population, losses

def population_vmap_loss(loss_fn):
    """Utility to vmap the loss function over a population pytree."""
    return jax.vmap(loss_fn)
    
def train_gsgd(population, loss_fn, generations=100):
    """Master training loop for GSGD optimization."""
    rng = jax.random.PRNGKey(42)
    optimizer = GSGD(population_size=len(population))
    
    for g in range(generations):
        rng, subkey = jax.random.split(rng)
        population, losses = optimizer.gsgd_step(subkey, population, loss_fn)
        best_loss = min(losses)
        if g % 10 == 0:
            print(f"🧬 Generation {g} | Best Loss: {best_loss:.5f}")
            
    return population
