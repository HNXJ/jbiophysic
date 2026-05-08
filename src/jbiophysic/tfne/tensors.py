"""Passive tensor parameterizations for TFNE."""

from __future__ import annotations

import jax
import jax.numpy as jnp

Array = jax.Array


def gamma_from_cholesky_params(L_params: Array, eps: float = 1e-8) -> Array:
    """Return SPD tensor field `Gamma = L L^T + eps I`.

    Parameters
    ----------
    L_params:
        Array with shape `[3, 3, Nx, Ny, Nz]`. Only the lower triangle is used.
    eps:
        Positive diagonal stabilizer in the same scale as `Gamma`.
    """
    if L_params.shape[:2] != (3, 3):
        raise ValueError("L_params must have leading shape (3, 3)")
    if eps <= 0:
        raise ValueError("eps must be positive")
    L = jnp.tril(L_params, k=0)
    gamma = jnp.einsum("ikxyz,jkxyz->ijxyz", L, L)
    eye = jnp.eye(3, dtype=L_params.dtype).reshape(3, 3, 1, 1, 1)
    return gamma + eps * eye


def isotropic_gamma(
    scalar_s_per_m: float,
    shape: tuple[int, int, int],
    dtype: jnp.dtype = jnp.float32,
) -> Array:
    """Create a spatially uniform isotropic conductivity/admittivity tensor field."""
    if scalar_s_per_m <= 0:
        raise ValueError("scalar_s_per_m must be positive")
    eye = jnp.eye(3, dtype=dtype).reshape(3, 3, 1, 1, 1)
    return scalar_s_per_m * jnp.broadcast_to(eye, (3, 3, *shape))


def tensor_eigenvalue_diagnostics(Gamma: Array) -> tuple[Array, Array, Array]:
    """Return `(min_eig, max_eig, condition_number)` over a tensor field."""
    if Gamma.shape[:2] != (3, 3):
        raise ValueError("Gamma must have leading shape (3, 3)")
    G = jnp.moveaxis(Gamma, (0, 1), (-2, -1))
    eigs = jnp.linalg.eigvalsh(G)
    min_eig = jnp.min(eigs)
    max_eig = jnp.max(eigs)
    cond = max_eig / jnp.maximum(min_eig, jnp.finfo(Gamma.dtype).tiny)
    return min_eig, max_eig, cond
