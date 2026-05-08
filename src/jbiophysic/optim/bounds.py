"""Optimizer bounds and parameter transforms."""

from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp


@dataclass(frozen=True)
class Bound:
    lower: float
    upper: float

    def __post_init__(self) -> None:
        if not self.lower < self.upper:
            raise ValueError("lower must be less than upper")

    def clip(self, x: jnp.ndarray) -> jnp.ndarray:
        return jnp.clip(x, self.lower, self.upper)


def sigmoid_bounded(u: jnp.ndarray, bound: Bound) -> jnp.ndarray:
    return bound.lower + (bound.upper - bound.lower) * jax_sigmoid(u)


def jax_sigmoid(u: jnp.ndarray) -> jnp.ndarray:
    return 1.0 / (1.0 + jnp.exp(-u))


def positive_softplus(u: jnp.ndarray, eps: float = 1e-8) -> jnp.ndarray:
    return jnp.logaddexp(u, 0.0) + eps
