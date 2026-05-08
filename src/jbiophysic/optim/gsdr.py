"""Genetic Stochastic Delta Rule primitives."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from .sdr import supervised_delta_direction


def gsdr_direction(
    key: jax.Array,
    theta_window: jnp.ndarray,
    loss_window: jnp.ndarray,
    theta_scale: jnp.ndarray,
    *,
    alpha: float = 0.5,
    eps: float = 1e-8,
) -> jnp.ndarray:
    """Blend supervised delta direction with genetic exploration."""
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be in [0, 1]")
    d_sup = supervised_delta_direction(theta_window, loss_window, eps=eps)
    noise = jax.random.normal(key, theta_scale.shape) * theta_scale
    d_gen = noise / (jnp.linalg.norm(noise) + eps)
    return (1.0 - alpha) * d_sup + alpha * d_gen
