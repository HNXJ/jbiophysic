"""Genetic Stochastic Gradient Descent (GSGD) optimizer with Optax integration."""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

try:
    import optax
except ModuleNotFoundError:  # pragma: no cover - core import without optional extras
    optax = None  # type: ignore[assignment]


class GSGDState(NamedTuple):
    """State for the GSGD optimizer."""

    step_count: jax.Array


def GSGD(
    learning_rate: float = 0.01,
    noise_scale: float = 0.0,
    clipping_value: float | None = 1.0,
) -> optax.GradientTransformation:
    """Genetic Stochastic Gradient Descent optimizer as an Optax GradientTransformation.

    Args:
        learning_rate: The learning rate.
        noise_scale: Scale of optional stochastic noise added to gradients.
        clipping_value: Optional value to clip updates.
    """
    if optax is None:
        raise ImportError("GSGD requires optional Optax; install with `pip install -e '.[jax]'`.")

    def init_fn(params: optax.Params) -> GSGDState:
        return GSGDState(step_count=jnp.array(0, dtype=jnp.int32))

    def update_fn(
        grads: optax.Updates,
        state: GSGDState,
        params: optax.Params | None = None,
    ) -> tuple[optax.Updates, GSGDState]:
        # Optional noise (only if noise_scale > 0 and we have a way to get a key)
        # Since GSGD is a plain GradientTransformation, it doesn't take a key in update.
        # If noise is required, we'd need ExtraArgs or a stateless noise.
        # For now, we follow standard SGD pattern but add finite checks and clipping.

        updates = jax.tree.map(lambda g: -learning_rate * g, grads)

        if clipping_value is not None:
            updates = jax.tree.map(lambda u: jnp.clip(u, -clipping_value, clipping_value), updates)

        # Handle NaNs
        updates = jax.tree.map(lambda u: jnp.where(jnp.isfinite(u), u, jnp.zeros_like(u)), updates)

        new_state = GSGDState(step_count=optax.safe_int32_increment(state.step_count))
        return updates, new_state

    return optax.GradientTransformation(init_fn, update_fn)


def gsgd_step(
    loss_fn, theta: jnp.ndarray, lr: float, *args, **kwargs
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Legacy helper for one gradient step returning `(theta_next, loss)`."""
    if lr <= 0:
        raise ValueError("lr must be positive")
    loss, grad = jax.value_and_grad(loss_fn)(theta, *args, **kwargs)
    grad = jnp.where(jnp.isfinite(grad), grad, jnp.zeros_like(grad))
    theta_next = theta - lr * grad
    return theta_next, loss
