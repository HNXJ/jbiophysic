"""Stochastic/delta-rule optimizer helper."""

from __future__ import annotations

import jax.numpy as jnp


def supervised_delta_direction(theta_window: jnp.ndarray, loss_window: jnp.ndarray, eps: float = 1e-8) -> jnp.ndarray:
    """History-derived descent heuristic; not a guaranteed gradient."""
    if theta_window.ndim != 2 or loss_window.ndim != 1:
        raise ValueError("theta_window must be [window, params], loss_window must be [window]")
    if theta_window.shape[0] != loss_window.shape[0]:
        raise ValueError("window lengths must match")
    centered_theta = theta_window - jnp.mean(theta_window, axis=0, keepdims=True)
    centered_loss = loss_window - jnp.mean(loss_window)
    d_bad = jnp.sum(centered_loss[:, None] * centered_theta, axis=0)
    return -d_bad / (jnp.linalg.norm(d_bad) + eps)
