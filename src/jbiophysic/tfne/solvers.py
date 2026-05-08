"""Small TFNE solver helpers.

These routines are intentionally smoke-test scale. Production elliptic solves should use a
proper gauge-fixed sparse solver or FEM backend.
"""

from __future__ import annotations

import jax.numpy as jnp

from .fields import TFNEGrid, mean_zero_gauge


def jacobi_poisson_neumann_smoke(
    source: jnp.ndarray,
    grid: TFNEGrid,
    *,
    conductivity_s_m: float = 0.3,
    steps: int = 200,
) -> jnp.ndarray:
    """Approximate `sigma * laplacian(phi) = -source` for small smoke tests.

    The source is mean-centered to satisfy a zero-flux Neumann compatibility condition.
    Only uniform spacing is supported.
    """
    if conductivity_s_m <= 0:
        raise ValueError("conductivity_s_m must be positive")
    if len(set(float(x) for x in grid.dx)) != 1:
        raise ValueError("smoke Jacobi solver requires equal grid spacing")
    if steps < 1:
        raise ValueError("steps must be positive")
    h = float(grid.dx[0])
    rhs = source - jnp.mean(source)
    phi = jnp.zeros(grid.shape, dtype=source.dtype)
    for _ in range(steps):
        p = jnp.pad(phi, ((1, 1), (1, 1), (1, 1)), mode="edge")
        neighbor_sum = (
            p[2:, 1:-1, 1:-1]
            + p[:-2, 1:-1, 1:-1]
            + p[1:-1, 2:, 1:-1]
            + p[1:-1, :-2, 1:-1]
            + p[1:-1, 1:-1, 2:]
            + p[1:-1, 1:-1, :-2]
        )
        phi = (neighbor_sum + (h**2) * rhs / conductivity_s_m) / 6.0
        phi = mean_zero_gauge(phi, grid.active_mask)
    return phi
