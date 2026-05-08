"""CSD and finite-difference field operators."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from .fields import TFNEGrid

Array = jax.Array


def _pad_edge(phi: Array) -> Array:
    return jnp.pad(phi, ((1, 1), (1, 1), (1, 1)), mode="edge")


def gradient_neumann_zero(phi: Array, grid: TFNEGrid) -> Array:
    """Central-difference gradient with edge padding; returns `[3, Nx, Ny, Nz]`."""
    dx, dy, dz = grid.dx
    p = _pad_edge(phi)
    gx = (p[2:, 1:-1, 1:-1] - p[:-2, 1:-1, 1:-1]) / (2.0 * dx)
    gy = (p[1:-1, 2:, 1:-1] - p[1:-1, :-2, 1:-1]) / (2.0 * dy)
    gz = (p[1:-1, 1:-1, 2:] - p[1:-1, 1:-1, :-2]) / (2.0 * dz)
    return jnp.stack([gx, gy, gz], axis=0)


def divergence_neumann_zero(J: Array, grid: TFNEGrid) -> Array:
    """Central-difference divergence for `J` with shape `[3, Nx, Ny, Nz]`."""
    dx, dy, dz = grid.dx
    Jx = _pad_edge(J[0])
    Jy = _pad_edge(J[1])
    Jz = _pad_edge(J[2])
    dJx = (Jx[2:, 1:-1, 1:-1] - Jx[:-2, 1:-1, 1:-1]) / (2.0 * dx)
    dJy = (Jy[1:-1, 2:, 1:-1] - Jy[1:-1, :-2, 1:-1]) / (2.0 * dy)
    dJz = (Jz[1:-1, 1:-1, 2:] - Jz[1:-1, 1:-1, :-2]) / (2.0 * dz)
    return dJx + dJy + dJz


def current_density(phi: Array, Gamma: Array, grid: TFNEGrid) -> Array:
    """Compute vector current density `J = -Gamma grad(phi)` in `A/m^2`."""
    grad_phi = gradient_neumann_zero(phi, grid)
    return -jnp.einsum("ijxyz,jxyz->ixyz", Gamma, grad_phi)


def elliptic_operator(phi: Array, Gamma: Array, grid: TFNEGrid) -> Array:
    """Compute `div(-Gamma grad phi)` for smoke-testable Poisson operators."""
    return divergence_neumann_zero(current_density(phi, Gamma, grid), grid)


def extracellular_csd(chi_per_m: Array, I_m_A_per_m2: Array, I_ext_A_per_m3: Array) -> Array:
    """Recommended sign convention: CSD = chi * I_m + I_ext."""
    return chi_per_m * I_m_A_per_m2 + I_ext_A_per_m3
