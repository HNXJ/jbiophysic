"""TFNE source mollification and conservation checks."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from .fields import TFNEGrid

Array = jax.Array


def gaussian_mollifier(grid: TFNEGrid, source_pos_m: Array, radius_m: float) -> Array:
    """Normalized Gaussian source kernel with units `1/m^3`.

    Discrete normalization enforces `sum(eta * voxel_volume) = 1` on active voxels.
    """
    if radius_m <= 0:
        raise ValueError("radius_m must be positive")
    source_pos_m = jnp.asarray(source_pos_m)
    if source_pos_m.shape != (3,):
        raise ValueError("source_pos_m must have shape (3,)")
    r2 = jnp.sum((grid.coords - source_pos_m) ** 2, axis=-1)
    raw = jnp.exp(-0.5 * r2 / (radius_m**2))
    raw = jnp.where(grid.active_mask, raw, 0.0)
    denom = jnp.sum(raw * grid.voxel_volume)
    denom = jnp.maximum(denom, jnp.finfo(raw.dtype).tiny)
    return raw / denom


def source_from_current(grid: TFNEGrid, current_A: Array, pos_m: Array, radius_m: float) -> Array:
    """Project a single current in amperes into a volumetric source field `A/m^3`."""
    return current_A * gaussian_mollifier(grid, pos_m, radius_m)


def project_sparse_currents(
    grid: TFNEGrid,
    currents_A: Array,
    positions_m: Array,
    radii_m: Array,
) -> Array:
    """Project sparse point-like currents to a conserved volumetric source field."""
    currents_A = jnp.asarray(currents_A)
    positions_m = jnp.asarray(positions_m)
    radii_m = jnp.asarray(radii_m)
    if currents_A.ndim != 1:
        raise ValueError("currents_A must be one-dimensional")
    if positions_m.shape != (currents_A.shape[0], 3):
        raise ValueError("positions_m must have shape (N, 3)")
    if radii_m.shape != currents_A.shape:
        raise ValueError("radii_m must have shape (N,)")

    q = jnp.zeros(grid.shape, dtype=currents_A.dtype)
    for current_val, pos, radius in zip(currents_A, positions_m, radii_m, strict=True):
        q = q + source_from_current(grid, current_val, pos, float(radius))
    return q


def integrate_source(grid: TFNEGrid, q_A_per_m3: Array) -> Array:
    """Integrate a volumetric source field over the grid in amperes."""
    return jnp.sum(jnp.where(grid.active_mask, q_A_per_m3, 0.0) * grid.voxel_volume)


def conservation_error(grid: TFNEGrid, q_A_per_m3: Array, target_A: Array) -> Array:
    """Return integrated source minus target current."""
    return integrate_source(grid, q_A_per_m3) - target_A
