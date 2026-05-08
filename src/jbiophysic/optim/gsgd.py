"""Small gradient-step helpers for JAX objective functions."""

from __future__ import annotations

import jax
import jax.numpy as jnp


def gsgd_step(loss_fn, theta: jnp.ndarray, lr: float, *args, **kwargs) -> tuple[jnp.ndarray, jnp.ndarray]:
    """One gradient step returning `(theta_next, loss)` for simple smoke tests."""
    if lr <= 0:
        raise ValueError("lr must be positive")
    loss, grad = jax.value_and_grad(loss_fn)(theta, *args, **kwargs)
    return theta - lr * grad, loss
