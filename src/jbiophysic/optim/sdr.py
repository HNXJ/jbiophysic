"""Stochastic Delta Rule (SDR) optimizer with Optax integration."""

from __future__ import annotations

from typing import Any, NamedTuple

import jax
import jax.numpy as jnp

try:  # Optax is optional for the core package; pure SDR helpers remain importable.
    import optax
except ModuleNotFoundError:  # pragma: no cover - exercised by core import smoke tests
    optax = None  # type: ignore[assignment]


class SDRState(NamedTuple):
    """State for the SDR optimizer."""

    momentum: optax.Updates
    step_count: jax.Array


def supervised_delta_direction(
    theta_window: jnp.ndarray, loss_window: jnp.ndarray, eps: float = 1e-8
) -> jnp.ndarray:
    """History-derived descent heuristic; not a guaranteed gradient."""
    if theta_window.ndim != 2 or loss_window.ndim != 1:
        raise ValueError("theta_window must be [window, params], loss_window must be [window]")
    if theta_window.shape[0] != loss_window.shape[0]:
        raise ValueError("window lengths must match")
    centered_theta = theta_window - jnp.mean(theta_window, axis=0, keepdims=True)
    centered_loss = loss_window - jnp.mean(loss_window)
    d_bad = jnp.sum(centered_loss[:, None] * centered_theta, axis=0)
    return -d_bad / (jnp.linalg.norm(d_bad) + eps)


def SDR(
    learning_rate: float = 0.01,
    momentum_beta: float = 0.9,
    stochastic_scale: float = 0.1,
    clipping_value: float | None = 1.0,
) -> optax.GradientTransformationExtraArgs:
    """Stochastic Delta Rule optimizer as an Optax GradientTransformationExtraArgs.

    Args:
        learning_rate: The learning rate.
        momentum_beta: The momentum decay factor.
        stochastic_scale: The scale of the stochastic updates.
        clipping_value: Optional value to clip updates.
    """
    if optax is None:
        raise ImportError("SDR requires optional Optax; install with `pip install -e '.[jax]'`.")

    def init_fn(params: optax.Params) -> SDRState:
        return SDRState(
            momentum=jax.tree.map(jnp.zeros_like, params),
            step_count=jnp.array(0, dtype=jnp.int32),
        )

    def update_fn(
        grads: optax.Updates,
        state: SDRState,
        params: optax.Params | None = None,
        *,
        key: jax.Array | None = None,
        **extra_args: Any,
    ) -> tuple[optax.Updates, SDRState]:
        if key is None:
            # Fallback for when key is not provided.
            # SDR technically requires it for stochasticity; we use a dummy key to stay JIT-safe.
            key = jax.random.PRNGKey(0)

        # Split key for tree-shaped updates
        flat_params, treedef = jax.tree.flatten(params)
        keys_flat = jax.random.split(key, len(flat_params))
        keys = jax.tree.unflatten(treedef, keys_flat)

        def _update_leaf(g, m, p, k):
            # Handle non-finite grads
            g_safe = jnp.where(jnp.isfinite(g), g, 0.0)

            # Stochastic delta rule: blend gradient with stochastic noise
            noise = jax.random.normal(k, g.shape) * stochastic_scale
            # We use the sign of the gradient as a descent direction
            d_descent = -jnp.sign(g_safe)

            # Update is only applied where g is finite (or we use g_safe)
            # If g was non-finite, d_descent is 0.
            update = learning_rate * (d_descent + noise)

            # If the original g was not finite, we might want to force update to 0
            # according to "omit them" requirement.
            update = jnp.where(jnp.isfinite(g), update, 0.0)

            # Update momentum
            new_m = momentum_beta * m + (1.0 - momentum_beta) * update
            return new_m, update

        # Apply updates leaf-wise
        momentum_and_updates = jax.tree.map(
            _update_leaf, grads, state.momentum, params, keys, is_leaf=None
        )
        new_momentum = jax.tree.map(
            lambda x: x[0], momentum_and_updates, is_leaf=lambda x: isinstance(x, tuple)
        )
        updates = jax.tree.map(
            lambda x: x[1], momentum_and_updates, is_leaf=lambda x: isinstance(x, tuple)
        )

        if clipping_value is not None:
            updates = jax.tree.map(lambda u: jnp.clip(u, -clipping_value, clipping_value), updates)

        # Handle NaNs in updates
        updates = jax.tree.map(lambda u: jnp.where(jnp.isfinite(u), u, jnp.zeros_like(u)), updates)

        new_state = SDRState(
            momentum=new_momentum,
            step_count=optax.safe_int32_increment(state.step_count),
        )
        return updates, new_state

    return optax.GradientTransformationExtraArgs(init_fn, update_fn)
