"""Tensor Field Neural Equation helpers under the `jbiophysic.tfne` namespace."""

from .fields import TFNEGrid, assert_finite_tree, initialize_potentials, make_regular_grid, mean_zero_gauge, pin_gauge
from .sources import conservation_error, gaussian_mollifier, integrate_source, project_sparse_currents, source_from_current
from .tensors import gamma_from_cholesky_params, isotropic_gamma, tensor_eigenvalue_diagnostics
from .csd import current_density, divergence_neumann_zero, elliptic_operator, extracellular_csd, gradient_neumann_zero
from .solvers import jacobi_poisson_neumann_smoke

__all__ = [
    "TFNEGrid",
    "assert_finite_tree",
    "initialize_potentials",
    "make_regular_grid",
    "mean_zero_gauge",
    "pin_gauge",
    "conservation_error",
    "gaussian_mollifier",
    "integrate_source",
    "project_sparse_currents",
    "source_from_current",
    "gamma_from_cholesky_params",
    "isotropic_gamma",
    "tensor_eigenvalue_diagnostics",
    "current_density",
    "divergence_neumann_zero",
    "elliptic_operator",
    "extracellular_csd",
    "gradient_neumann_zero",
    "jacobi_poisson_neumann_smoke",
]
