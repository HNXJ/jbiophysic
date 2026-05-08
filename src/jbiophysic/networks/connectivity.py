"""Connectivity matrix helpers."""

from __future__ import annotations

import jax
import jax.numpy as jnp


def erdos_renyi_mask(key: jax.Array, n_pre: int, n_post: int, p: float, *, allow_self: bool = False) -> jnp.ndarray:
    """Return a boolean random connectivity mask."""
    if not 0.0 <= p <= 1.0:
        raise ValueError("p must be in [0, 1]")
    mask = jax.random.bernoulli(key, p, shape=(n_pre, n_post))
    if n_pre == n_post and not allow_self:
        mask = mask & ~jnp.eye(n_pre, dtype=bool)
    return mask


def weight_matrix(mask: jnp.ndarray, weight: float) -> jnp.ndarray:
    """Convert a connectivity mask to a fixed-weight matrix."""
    return jnp.where(mask, weight, 0.0)
