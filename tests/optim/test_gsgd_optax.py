"""Tests for the GSGD Optax optimizer."""

import pytest

pytest.importorskip("optax")

import jax
import jax.numpy as jnp

from jbiophysic.optim.gsgd import GSGD, gsgd_step


def test_gsgd_optax_compliance():
    tx = GSGD(learning_rate=0.01)
    params = jnp.array([1.0, 2.0])
    state = tx.init(params)

    assert state.step_count == 0

    grads = jnp.array([0.1, -0.1])
    updates, next_state = tx.update(grads, state, params)

    assert next_state.step_count == 1
    assert jnp.allclose(updates, -0.01 * grads)


def test_gsgd_step_legacy():
    def loss_fn(x):
        return jnp.sum(x**2)

    theta = jnp.array([1.0, 2.0])
    theta_next, loss = gsgd_step(loss_fn, theta, 0.1)

    assert float(loss) == 5.0
    # grad is [2.0, 4.0]
    # theta_next = [1.0 - 0.1*2.0, 2.0 - 0.1*4.0] = [0.8, 1.6]
    assert jnp.allclose(theta_next, jnp.array([0.8, 1.6]))


def test_gsgd_jit():
    tx = GSGD(learning_rate=0.01)
    params = jnp.array([1.0, 2.0])
    state = tx.init(params)

    @jax.jit
    def step(p, s):
        g = p * 0.1
        return tx.update(g, s, p)

    updates, next_state = step(params, state)
    assert next_state.step_count == 1
