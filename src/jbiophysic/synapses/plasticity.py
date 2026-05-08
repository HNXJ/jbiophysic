"""Minimal differentiable plasticity helpers."""

from __future__ import annotations

import jax.numpy as jnp


def mcdp_correlation(pre: jnp.ndarray, post: jnp.ndarray, eps: float = 1e-8) -> jnp.ndarray:
    """Mutual-correlation dependent plasticity factor from time x neurons signals."""
    if pre.ndim != 2 or post.ndim != 2:
        raise ValueError("pre and post must be [time, neurons] arrays")
    pre_z = (pre - jnp.mean(pre, axis=0, keepdims=True)) / (jnp.std(pre, axis=0, keepdims=True) + eps)
    post_z = (post - jnp.mean(post, axis=0, keepdims=True)) / (jnp.std(post, axis=0, keepdims=True) + eps)
    corr = pre_z.T @ post_z / jnp.maximum(pre.shape[0] - 1, 1)
    return (corr - jnp.mean(corr)) / (jnp.std(corr) + eps)
