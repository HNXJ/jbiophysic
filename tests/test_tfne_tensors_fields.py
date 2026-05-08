import jax.numpy as jnp

from jbiophysic.tfne.csd import current_density, elliptic_operator, extracellular_csd
from jbiophysic.tfne.fields import initialize_potentials, make_regular_grid, mean_zero_gauge
from jbiophysic.tfne.tensors import (
    gamma_from_cholesky_params,
    isotropic_gamma,
    tensor_eigenvalue_diagnostics,
)
from jbiophysic.tfne.validation import assert_no_nan_inf, assert_passive_spd


def test_spd_tensor_parameterization_is_positive():
    L = jnp.zeros((3, 3, 3, 3, 3))
    L = L.at[0, 0].set(0.3)
    L = L.at[1, 1].set(0.4)
    L = L.at[2, 2].set(0.5)
    gamma = gamma_from_cholesky_params(L, eps=1e-6)
    min_eig, max_eig, cond = tensor_eigenvalue_diagnostics(gamma)
    assert float(min_eig) > 0.0
    assert float(max_eig) > float(min_eig)
    assert float(cond) > 1.0
    assert_passive_spd(gamma)


def test_gauge_and_potential_initialization_are_consistent():
    phi_e = jnp.ones((4, 4, 4)) * 50e-6
    v_m = jnp.ones((4, 4, 4)) * -70e-3
    phi_i, phi_e_g, v_m_out = initialize_potentials(phi_e, v_m, active_mask=jnp.ones((4, 4, 4), dtype=bool))
    assert abs(float(jnp.mean(phi_e_g))) < 1e-10
    assert float(jnp.max(jnp.abs((phi_i - phi_e_g) - v_m_out))) < 1e-10


def test_csd_and_current_density_are_finite():
    grid = make_regular_grid((5, 5, 5), (1e-4, 1e-4, 1e-4))
    gamma = isotropic_gamma(0.3, grid.shape)
    phi = mean_zero_gauge(jnp.linspace(-1e-3, 1e-3, 125).reshape(grid.shape), grid.active_mask)
    J = current_density(phi, gamma, grid)
    op = elliptic_operator(phi, gamma, grid)
    csd = extracellular_csd(jnp.ones(grid.shape) * 1000.0, jnp.ones(grid.shape) * 1e-6, jnp.zeros(grid.shape))
    assert_no_nan_inf("J", J)
    assert_no_nan_inf("op", op)
    assert_no_nan_inf("csd", csd)
