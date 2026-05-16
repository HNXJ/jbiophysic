"""Adaptive Genetic Stochastic Delta Rule (AGSDR) optimizer with Optax integration."""

from __future__ import annotations

from typing import Any, NamedTuple

import jax
import jax.numpy as jnp

try:
    import optax
except ModuleNotFoundError:  # pragma: no cover - core import without optional extras
    optax = None  # type: ignore[assignment]

from .gsdr import GSDR, GSDRState


class AGSDRSchedule(NamedTuple):
    """Schedule for adaptive alpha in AGSDR."""

    alpha_min: float = 0.05
    alpha_max: float = 0.8
    alpha_up: float = 0.05
    alpha_down: float = 0.01


def adapt_alpha(
    alpha: jax.Array,
    *,
    plateau: jax.Array,
    improving: jax.Array,
    schedule: AGSDRSchedule | None = None,
) -> jax.Array:
    """Update alpha based on plateau/improvement flags (JIT-safe)."""
    if schedule is None:
        schedule = AGSDRSchedule()
    new_alpha = jnp.where(plateau, alpha + schedule.alpha_up, alpha)
    new_alpha = jnp.where(improving, new_alpha - schedule.alpha_down, new_alpha)
    return jnp.clip(new_alpha, schedule.alpha_min, schedule.alpha_max)


def AGSDR(
    inner_optimizer: optax.GradientTransformation,
    alpha: float = 0.5,
    lambda_d: float = 0.1,
    deselection_threshold: float = 1.5,
    plateau_threshold: int = 100,
    tau_a_growth: float = 50.0,
    alpha_schedule: AGSDRSchedule | None = None,
    clipping_value: float | None = 1.0,
) -> optax.GradientTransformationExtraArgs:
    """Adaptive GSDR optimizer.

    Args:
        inner_optimizer: The base optimizer.
        alpha: Initial exploration blending factor.
        lambda_d: Stochastic update scale.
        deselection_threshold: Loss ratio to trigger reset.
        plateau_threshold: Consecutive steps without improvement to trigger reset.
        tau_a_growth: Time constant for lambda_d growth.
        alpha_schedule: Schedule for adapting alpha.
        clipping_value: Optional value to clip updates.
    """
    if optax is None:
        raise ImportError("AGSDR requires optional Optax; install with `pip install -e '.[jax]'`.")
    if alpha_schedule is None:
        alpha_schedule = AGSDRSchedule()

    gsdr = GSDR(
        inner_optimizer=inner_optimizer,
        alpha=alpha,
        lambda_d=lambda_d,
        deselection_threshold=deselection_threshold,
        plateau_threshold=plateau_threshold,
        tau_a_growth=tau_a_growth,
        clipping_value=clipping_value,
    )

    def init_fn(params: optax.Params) -> GSDRState:
        return gsdr.init(params)

    def update_fn(
        grads: optax.Updates,
        state: GSDRState,
        params: optax.Params | None = None,
        *,
        value: jax.Array | None = None,
        key: jax.Array | None = None,
        **extra_args: Any,
    ) -> tuple[optax.Updates, GSDRState]:
        # Perform standard GSDR update
        updates, next_state = gsdr.update(grads, state, params, value=value, key=key, **extra_args)

        # Adapt alpha based on current performance
        is_improving = value < state.loss_opt
        is_finite = jnp.isfinite(value)
        is_valid_improvement = jnp.logical_and(is_improving, is_finite)

        # We consider it a plateau if it hasn't improved for a while
        # Note: next_state already has updated consecutive_unchanged_epochs
        is_plateau = next_state.consecutive_unchanged_epochs > 0

        new_a = adapt_alpha(
            next_state.a,
            plateau=is_plateau,
            improving=is_valid_improvement,
            schedule=alpha_schedule,
        )

        # Update alpha in the state
        final_state = next_state._replace(a=new_a)

        return updates, final_state

    return optax.GradientTransformationExtraArgs(init_fn, update_fn)
