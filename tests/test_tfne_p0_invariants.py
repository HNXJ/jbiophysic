import jax.numpy as jnp

from jbiophysic.tfne.csd import divergence_neumann_zero
from jbiophysic.tfne.fields import make_regular_grid, mean_zero_gauge, pin_gauge
from jbiophysic.tfne.sources import gaussian_mollifier, integrate_source, source_from_current
from jbiophysic.tfne.tensors import gamma_from_cholesky_params, tensor_eigenvalue_diagnostics


def test_spd_tensor_positive_eigenvalues():
    Nx, Ny, Nz = 4, 4, 4
    L_params = jnp.zeros((3, 3, Nx, Ny, Nz))
    eps = 1e-5
    Gamma = gamma_from_cholesky_params(L_params, eps=eps)
    min_eig, max_eig, cond = tensor_eigenvalue_diagnostics(Gamma)
    assert min_eig >= eps - 1e-7
    assert max_eig >= eps - 1e-7

def test_mean_zero_gauge_removes_constant_offset():
    Nx, Ny, Nz = 4, 4, 4
    phi = jnp.ones((Nx, Ny, Nz)) * 10.0
    active_mask = jnp.ones((Nx, Ny, Nz), dtype=bool)
    phi_fixed = mean_zero_gauge(phi, active_mask)
    assert jnp.allclose(jnp.mean(phi_fixed), 0.0, atol=1e-6)

def test_pin_gauge_sets_reference_value():
    grid = make_regular_grid((4, 4, 4), (0.1, 0.1, 0.1))
    phi = jnp.ones((4, 4, 4)) * 5.0
    phi_fixed = pin_gauge(phi, grid, value=0.0)
    i, j, k = grid.gauge_index
    assert phi_fixed[i, j, k] == 0.0

def test_mollifier_volume_integral_is_one():
    grid = make_regular_grid((10, 10, 10), (0.1, 0.1, 0.1)) # 1m^3 box, 10x10x10 = 1000 voxels
    # voxel_volume = 0.1 * 0.1 * 0.1 = 0.001 m^3
    source_pos = jnp.array([0.5, 0.5, 0.5])
    radius = 0.1
    eta = gaussian_mollifier(grid, source_pos, radius)
    total_vol_integral = jnp.sum(eta * grid.voxel_volume)
    assert jnp.allclose(total_vol_integral, 1.0, atol=1e-6)

def test_sparse_current_projection_conserves_current():
    grid = make_regular_grid((10, 10, 10), (0.1, 0.1, 0.1))
    current_A = jnp.array(5.0)
    pos = jnp.array([0.5, 0.5, 0.5])
    radius = 0.1
    q = source_from_current(grid, current_A, pos, radius)
    integrated = integrate_source(grid, q)
    assert jnp.allclose(integrated, 5.0, atol=1e-5)

def test_divergence_csd_sign_convention():
    # Simple check that divergence of a constant field is zero
    grid = make_regular_grid((4, 4, 4), (1.0, 1.0, 1.0))
    J = jnp.ones((3, 4, 4, 4))
    div_J = divergence_neumann_zero(J, grid)
    assert jnp.allclose(div_J, 0.0, atol=1e-6)
