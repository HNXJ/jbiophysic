"""Tests for the AGSDR Optax optimizer."""

import jax
import jax.numpy as jnp
import optax

from jbiophysic.optim.agsdr import AGSDR, AGSDRSchedule


def test_agsdr_alpha_adaptation():
    inner = optax.adam(0.01)
    schedule = AGSDRSchedule(alpha_min=0.1, alpha_max=0.8, alpha_up=0.1, alpha_down=0.05)
    tx = AGSDR(inner_optimizer=inner, alpha=0.5, alpha_schedule=schedule)
    
    params = jnp.array([1.0, 2.0])
    state = tx.init(params)
    assert state.a == 0.5
    
    key = jax.random.PRNGKey(0)
    
    # Case 1: Improvement -> Alpha should decrease
    _, state = tx.update(jnp.zeros(2), state, params, value=jnp.array(10.0), key=key)
    # Next step with improvement
    _, state = tx.update(jnp.zeros(2), state, params, value=jnp.array(5.0), key=key)
    
    # After two improvements, alpha should be 0.5 - 0.05 - 0.05 = 0.4
    assert jnp.allclose(state.a, 0.4)
    
    # Case 2: Plateau (no improvement) -> Alpha should increase
    # Current loss_opt is 5.0. Provide loss 6.0
    _, state = tx.update(jnp.zeros(2), state, params, value=jnp.array(6.0), key=key)
    # Alpha should increase: 0.4 + 0.1 = 0.5
    assert jnp.allclose(state.a, 0.5)


def test_agsdr_jit():
    inner = optax.adam(0.01)
    tx = AGSDR(inner_optimizer=inner)
    params = jnp.array([1.0, 2.0])
    state = tx.init(params)
    
    @jax.jit
    def step(p, s, v, k):
        g = p * 0.1
        return tx.update(g, s, p, value=v, key=k)
    
    key = jax.random.PRNGKey(0)
    updates, next_state = step(params, state, jnp.array(1.0), key)
    assert next_state.step_count == 1
