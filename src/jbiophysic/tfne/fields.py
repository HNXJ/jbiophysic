"""TFNE field and grid primitives.

All distances are SI meters and all potentials are SI volts unless a function explicitly
states otherwise. `phi_e` is gauge dependent; use `mean_zero_gauge` or `pin_gauge` before
interpreting extracellular potentials.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

Array = jax.Array


class TFNEGrid(NamedTuple):
    """Regular Cartesian grid for small TFNE smoke tests."""

    shape: tuple[int, int, int]
    dx: tuple[float, float, float]
    coords: Array
    voxel_volume: float
    active_mask: Array
    gauge_index: tuple[int, int, int] = (0, 0, 0)


def make_regular_grid(
    shape: tuple[int, int, int],
    dx: tuple[float, float, float],
    gauge_index: tuple[int, int, int] = (0, 0, 0),
) -> TFNEGrid:
    """Create a regular 3-D grid with coordinates in meters."""
    if len(shape) != 3 or len(dx) != 3:
        raise ValueError("shape and dx must each have length 3")
    if any(n <= 0 for n in shape):
        raise ValueError("grid shape entries must be positive")
    if any(step <= 0 for step in dx):
        raise ValueError("grid spacing entries must be positive")

    x = jnp.arange(shape[0]) * dx[0]
    y = jnp.arange(shape[1]) * dx[1]
    z = jnp.arange(shape[2]) * dx[2]
    X, Y, Z = jnp.meshgrid(x, y, z, indexing="ij")
    coords = jnp.stack([X, Y, Z], axis=-1)
    return TFNEGrid(
        shape=shape,
        dx=dx,
        coords=coords,
        voxel_volume=float(dx[0] * dx[1] * dx[2]),
        active_mask=jnp.ones(shape, dtype=bool),
        gauge_index=gauge_index,
    )


def mean_zero_gauge(phi: Array, active_mask: Array) -> Array:
    """Set the mean potential over active voxels to zero."""
    denom = jnp.maximum(jnp.sum(active_mask), 1)
    mu = jnp.sum(jnp.where(active_mask, phi, 0.0)) / denom
    return jnp.where(active_mask, phi - mu, phi)


def pin_gauge(phi: Array, grid: TFNEGrid, value: float = 0.0) -> Array:
    """Set one grid node to a reference value to remove additive gauge freedom."""
    i, j, k = grid.gauge_index
    return phi.at[i, j, k].set(value)


def initialize_potentials(
    phi_e: Array,
    v_m: Array,
    *,
    active_mask: Array | None = None,
) -> tuple[Array, Array, Array]:
    """Return algebraically consistent `(phi_i, phi_e, V_m)`.

    TFNE defines `V_m = phi_i - phi_e`; do not independently sample all three fields.
    """
    if phi_e.shape != v_m.shape:
        raise ValueError("phi_e and v_m must have the same shape")
    if active_mask is not None:
        phi_e = mean_zero_gauge(phi_e, active_mask)
    phi_i = v_m + phi_e
    return phi_i, phi_e, v_m


def assert_finite_tree(*arrays: Array) -> None:
    """Raise `FloatingPointError` if any supplied array contains NaN/Inf."""
    for arr in arrays:
        if not bool(jnp.all(jnp.isfinite(arr))):
            raise FloatingPointError("non-finite value detected")
