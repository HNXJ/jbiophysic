"""Tests for JIT and pmap compatibility of optimizers."""

import pytest

pytest.importorskip("optax")

import jax
import jax.numpy as jnp
import optax
import pytest

from jbiophysic.optim.gsdr import GSDR


def test_gsdr_pmap_smoke():
    """Smoke test for pmap compatibility."""
    try:
        devices = jax.local_device_count()
    except Exception:
        pytest.skip("No JAX devices available for pmap smoke")

    if devices < 1:
        pytest.skip("No JAX devices available")

    inner = optax.adam(0.01)
    tx = GSDR(inner_optimizer=inner)

    params = jnp.array([1.0, 2.0])
    state = tx.init(params)

    # Replicate for pmap
    params_repl = jax.tree.map(lambda x: jnp.stack([x] * devices), params)
    state_repl = jax.tree.map(lambda x: jnp.stack([x] * devices), state)
    keys = jax.random.split(jax.random.PRNGKey(0), devices)
    losses = jnp.array([1.0] * devices)

    @jax.pmap
    def step(p, s, v, k):
        g = p * 0.1
        return tx.update(g, s, p, value=v, key=k)

    updates, next_state = step(params_repl, state_repl, losses, keys)

    # Verify we got replicated outputs
    assert updates.shape == (devices, 2)
    assert next_state.step_count.shape == (devices,)
    assert jnp.all(next_state.step_count == 1)


def test_gsdr_vmap_smoke():
    """Smoke test for vmap compatibility (e.g., batched losses)."""
    inner = optax.adam(0.01)
    tx = GSDR(inner_optimizer=inner)

    params = jnp.array([1.0, 2.0])
    state = tx.init(params)

    # We want to vmap over different keys and losses for the same params/state
    keys = jax.random.split(jax.random.PRNGKey(0), 4)
    losses = jnp.array([1.0, 0.5, 2.0, 1.5])

    @jax.vmap
    def batched_update(k, v):
        g = params * 0.1
        return tx.update(g, state, params, value=v, key=k)

    updates, next_state = batched_update(keys, losses)

    assert updates.shape == (4, 2)
    assert next_state.step_count.shape == (4,)
