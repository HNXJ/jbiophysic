"""Tests for biophysical objectives."""

import jax
import jax.numpy as jnp

from jbiophysic.objectives.synchrony import synchrony_kappa_objective


def test_synchrony_kappa_objective():
    # Deterministic signals
    time = jnp.linspace(0, 10, 100)
    # Identical signals -> High kappa
    v1 = jnp.sin(time)[:, None]
    v_traces_sync = jnp.concatenate([v1, v1, v1], axis=1)
    
    loss_sync = synchrony_kappa_objective(v_traces_sync, target_kappa=0.0)
    
    # Orthogonal signals (approx) -> Low kappa
    v2 = jnp.cos(time)[:, None]
    v3 = jnp.sin(2*time)[:, None]
    v_traces_async = jnp.concatenate([v1, v2, v3], axis=1)
    
    loss_async = synchrony_kappa_objective(v_traces_async, target_kappa=0.0)
    
    # Async should have lower loss when target is 0
    assert loss_async < loss_sync


def test_synchrony_no_silence_penalty():
    # Flat signals -> Silence
    v_traces_silent = jnp.zeros((100, 3))
    
    # High threshold to trigger penalty
    loss_silent = synchrony_kappa_objective(
        v_traces_silent, no_silence_threshold=1.0, no_silence_weight=100.0
    )
    
    # Active signals
    v_traces_active = jax.random.normal(jax.random.PRNGKey(0), (100, 3))
    loss_active = synchrony_kappa_objective(
        v_traces_active, no_silence_threshold=0.1, no_silence_weight=100.0
    )
    
    assert loss_silent > loss_active


def test_synchrony_differentiable():
    def simple_loss(w):
        v = jnp.ones((10, 2)) * w
        return synchrony_kappa_objective(v)
    
    grad = jax.grad(simple_loss)(1.0)
    assert jnp.isfinite(grad)
