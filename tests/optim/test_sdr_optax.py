"""Tests for the SDR Optax optimizer."""

import pytest

pytest.importorskip("optax")

import jax
import jax.numpy as jnp

from jbiophysic.optim.sdr import SDR


def test_sdr_optax_compliance():
    tx = SDR(learning_rate=0.01)
    params = {"w": jnp.array([1.0, 2.0]), "b": jnp.array(0.0)}
    state = tx.init(params)

    # Check state structure
    assert "momentum" in state._fields
    assert "step_count" in state._fields
    assert state.step_count == 0

    grads = {"w": jnp.array([0.1, -0.1]), "b": jnp.array(0.05)}
    key = jax.random.PRNGKey(42)

    updates, next_state = tx.update(grads, state, params, key=key)

    assert next_state.step_count == 1
    assert jax.tree.all(jax.tree.map(lambda x: jnp.all(jnp.isfinite(x)), updates))


def test_sdr_jit():
    tx = SDR(learning_rate=0.01)
    params = jnp.array([1.0, 2.0])
    state = tx.init(params)

    @jax.jit
    def step(p, s, k):
        g = p * 0.1  # dummy grad
        return tx.update(g, s, p, key=k)

    key = jax.random.PRNGKey(0)
    updates, next_state = step(params, state, key)
    assert next_state.step_count == 1


def test_sdr_finite_updates():
    tx = SDR(learning_rate=0.01)
    params = jnp.array([1.0, 2.0])
    state = tx.init(params)

    # NaN grads should result in zero updates (or finite)
    grads = jnp.array([jnp.nan, jnp.inf])
    updates, next_state = tx.update(grads, state, params, key=jax.random.PRNGKey(0))

    assert jnp.all(jnp.isfinite(updates))
    assert jnp.all(updates == 0.0)
