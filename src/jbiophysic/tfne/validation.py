"""Validation helpers for TFNE smoke tests."""

from __future__ import annotations

import jax.numpy as jnp

from .fields import TFNEGrid
from .sources import integrate_source
from .tensors import tensor_eigenvalue_diagnostics


def assert_no_nan_inf(name: str, arr: jnp.ndarray) -> None:
    if not bool(jnp.all(jnp.isfinite(arr))):
        raise AssertionError(f"{name} contains NaN/Inf")


def assert_source_conserved(
    grid: TFNEGrid,
    q_A_per_m3: jnp.ndarray,
    target_A: float,
    *,
    rtol: float = 1e-4,
    atol: float = 1e-15,
) -> None:
    got = float(integrate_source(grid, q_A_per_m3))
    if abs(got - target_A) > atol + rtol * max(abs(target_A), atol):
        raise AssertionError(f"source conservation failed: got={got}, target={target_A}")


def assert_passive_spd(Gamma: jnp.ndarray, *, min_eig_floor: float = 0.0) -> None:
    min_eig, _max_eig, cond = tensor_eigenvalue_diagnostics(Gamma)
    if float(min_eig) <= min_eig_floor:
        raise AssertionError(f"tensor is not sufficiently SPD: min_eig={float(min_eig)}")
    if not bool(jnp.isfinite(cond)):
        raise AssertionError("tensor condition number is not finite")
