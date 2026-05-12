"""Tests for the GSDR Optax optimizer."""

import jax
import jax.numpy as jnp
import optax

from jbiophysic.optim.gsdr import GSDR


def test_gsdr_basic_flow():
    inner = optax.adam(0.01)
    tx = GSDR(inner_optimizer=inner, alpha=0.5)
    params = jnp.array([1.0, 2.0])
    state = tx.init(params)
    
    assert state.step_count == 0
    assert jnp.isinf(state.loss_opt)

    grads = jnp.array([0.1, -0.1])
    key = jax.random.PRNGKey(0)
    loss = jnp.array(10.0)
    
    updates, next_state = tx.update(grads, state, params, value=loss, key=key)
    
    assert next_state.step_count == 1
    assert next_state.loss_opt == 10.0
    assert jnp.all(next_state.params_opt == params)


def test_gsdr_improvement():
    inner = optax.adam(0.01)
    tx = GSDR(inner_optimizer=inner)
    params = jnp.array([1.0, 2.0])
    state = tx.init(params)
    
    key = jax.random.PRNGKey(0)
    
    # Step 1: Initial loss
    updates, state = tx.update(jnp.zeros(2), state, params, value=jnp.array(10.0), key=key)
    
    # Step 2: Improved loss
    new_params = params + updates
    updates, next_state = tx.update(jnp.zeros(2), state, new_params, value=jnp.array(5.0), key=key)
    
    assert next_state.loss_opt == 5.0
    assert jnp.all(next_state.params_opt == new_params)


def test_gsdr_deselection():
    inner = optax.adam(0.01)
    # Low threshold to trigger reset easily, disable clipping to see full reset
    tx = GSDR(inner_optimizer=inner, deselection_threshold=1.1, clipping_value=None)
    params = jnp.array([1.0, 2.0])
    state = tx.init(params)
    
    key = jax.random.PRNGKey(0)
    
    # Step 1: Establish loss_opt = 1.0
    _, state = tx.update(jnp.zeros(2), state, params, value=jnp.array(1.0), key=key)
    
    # Step 2: Much worse loss (2.0 > 1.0 * 1.1)
    bad_params = params + 5.0 
    updates, next_state = tx.update(jnp.zeros(2), state, bad_params, value=jnp.array(2.0), key=key)
    
    # Update should be (params_opt - bad_params) to reset
    expected_reset = params - bad_params
    assert jnp.allclose(updates, expected_reset)


def test_gsdr_jit():
    inner = optax.adam(0.01)
    tx = GSDR(inner_optimizer=inner)
    params = jnp.array([1.0, 2.0])
    state = tx.init(params)
    
    @jax.jit
    def step(p, s, v, k):
        g = p * 0.1
        return tx.update(g, s, p, value=v, key=k)
    
    key = jax.random.PRNGKey(0)
    updates, next_state = step(params, state, jnp.array(1.0), key)
    assert next_state.step_count == 1
