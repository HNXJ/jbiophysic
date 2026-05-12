"""Genetic Stochastic Delta Rule (GSDR) optimizer with Optax integration."""

from __future__ import annotations

from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import optax

from .sdr import supervised_delta_direction


class GSDRState(NamedTuple):
    """State for the GSDR optimizer."""

    inner_state: Any
    params_opt: optax.Params
    inner_state_opt: Any
    loss_opt: jax.Array
    a: jax.Array
    a_opt: jax.Array
    lambda_d: jax.Array
    step_count: jax.Array
    consecutive_unchanged_epochs: jax.Array
    last_optimal_change_step: jax.Array


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


def GSDR(
    inner_optimizer: optax.GradientTransformation,
    alpha: float = 0.5,
    lambda_d: float = 0.1,
    deselection_threshold: float = 1.5,
    plateau_threshold: int = 100,
    tau_a_growth: float = 50.0,
    clipping_value: float | None = 1.0,
) -> optax.GradientTransformationExtraArgs:
    """Genetic Stochastic Delta Rule optimizer as an Optax GradientTransformationExtraArgs.

    Args:
        inner_optimizer: The base optimizer (e.g., Adam, SGD).
        alpha: Exploration blending factor [0, 1].
        lambda_d: Stochastic update scale.
        deselection_threshold: Loss ratio (loss / loss_opt) to trigger reset.
        plateau_threshold: Consecutive steps without improvement to trigger reset.
        tau_a_growth: Time constant for bounded lambda_d growth.
        clipping_value: Optional value to clip updates.
    """

    def init_fn(params: optax.Params) -> GSDRState:
        return GSDRState(
            inner_state=inner_optimizer.init(params),
            params_opt=params,
            inner_state_opt=inner_optimizer.init(params),
            loss_opt=jnp.array(jnp.inf),
            a=jnp.array(alpha),
            a_opt=jnp.array(alpha),
            lambda_d=jnp.array(lambda_d),
            step_count=jnp.array(0, dtype=jnp.int32),
            consecutive_unchanged_epochs=jnp.array(0, dtype=jnp.int32),
            last_optimal_change_step=jnp.array(0, dtype=jnp.int32),
        )

    def update_fn(
        grads: optax.Updates,
        state: GSDRState,
        params: optax.Params | None = None,
        *,
        value: jax.Array | None = None,
        key: jax.Array | None = None,
        **extra_args: Any,
    ) -> tuple[optax.Updates, GSDRState]:
        if params is None:
            raise ValueError("GSDR requires params in update.")
        if value is None:
            raise ValueError("GSDR requires value (loss) in update.")
        if key is None:
            key = jax.random.PRNGKey(0)

        # Check for improvement
        is_improving = value < state.loss_opt
        is_finite = jnp.isfinite(value)
        is_valid_improvement = jnp.logical_and(is_improving, is_finite)

        # Update best known state
        params_opt = jax.tree.map(
            lambda p_opt, p: jnp.where(is_valid_improvement, p, p_opt), state.params_opt, params
        )
        loss_opt = jnp.where(is_valid_improvement, value, state.loss_opt)
        inner_state_opt = jax.tree.map(
            lambda s_opt, s: jnp.where(is_valid_improvement, s, s_opt),
            state.inner_state_opt,
            state.inner_state,
        )
        last_optimal_change_step = jnp.where(
            is_valid_improvement, state.step_count, state.last_optimal_change_step
        )
        consecutive_unchanged_epochs = jnp.where(
            is_valid_improvement, 0, state.consecutive_unchanged_epochs + 1
        )

        # Deselection logic
        loss_ratio = value / (loss_opt + 1e-8)
        too_bad = loss_ratio > deselection_threshold
        is_plateau = consecutive_unchanged_epochs > plateau_threshold
        bad_or_plateau = jnp.logical_or(too_bad, is_plateau)
        should_reset = jnp.logical_or(bad_or_plateau, jnp.logical_not(is_finite))

        # Inner optimizer update
        inner_updates, inner_state = inner_optimizer.update(
            grads, state.inner_state, params, **extra_args
        )

        # Stochastic Exploration update
        flat_params, treedef = jax.tree.flatten(params)
        keys_flat = jax.random.split(key, len(flat_params))
        keys = jax.tree.unflatten(treedef, keys_flat)

        time_since_last_change = (state.step_count - last_optimal_change_step).astype(jnp.float32)
        decay_factor = jnp.exp(-time_since_last_change / tau_a_growth)
        effective_lambda_d = state.lambda_d * (1.0 - decay_factor)

        def _stochastic_update(p, k):
            noise = jax.random.normal(k, p.shape)
            # Use alpha for blending if desired, here we use lambda_d for scale
            return noise * effective_lambda_d

        stochastic_updates = jax.tree.map(_stochastic_update, params, keys)

        # Combine updates
        def _combine(i_u, s_u, p, p_opt):
            # Normal step: inner + stochastic
            normal_update = (1.0 - state.a) * i_u + state.a * s_u
            # Reset step: jump back to params_opt
            reset_update = p_opt - p
            return jnp.where(should_reset, reset_update, normal_update)

        updates = jax.tree.map(_combine, inner_updates, stochastic_updates, params, params_opt)

        if clipping_value is not None:
            updates = jax.tree.map(lambda u: jnp.clip(u, -clipping_value, clipping_value), updates)

        # Handle NaNs
        updates = jax.tree.map(lambda u: jnp.where(jnp.isfinite(u), u, jnp.zeros_like(u)), updates)

        new_state = GSDRState(
            inner_state=inner_state,
            params_opt=params_opt,
            inner_state_opt=inner_state_opt,
            loss_opt=loss_opt,
            a=state.a,
            a_opt=state.a_opt,
            lambda_d=state.lambda_d,
            step_count=optax.safe_int32_increment(state.step_count),
            consecutive_unchanged_epochs=consecutive_unchanged_epochs,
            last_optimal_change_step=last_optimal_change_step,
        )

        return updates, new_state

    return optax.GradientTransformationExtraArgs(init_fn, update_fn)
