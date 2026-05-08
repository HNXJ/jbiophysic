"""JAX-native biophysical modeling and TFNE science-library kernels."""

from .cells.hh import HHParams, simulate_hh
from .cells.izhikevich import IzhikevichParams, simulate_izhikevich
from .networks.cortex import CortexNetworkSpec, make_cortex_network, make_cortex_network_json
from .tfne.fields import TFNEGrid, make_regular_grid
from .tfne.sources import conservation_error, gaussian_mollifier, project_sparse_currents
from .tfne.tensors import gamma_from_cholesky_params, tensor_eigenvalue_diagnostics

__all__ = [
    "IzhikevichParams",
    "simulate_izhikevich",
    "HHParams",
    "simulate_hh",
    "TFNEGrid",
    "make_regular_grid",
    "gaussian_mollifier",
    "project_sparse_currents",
    "conservation_error",
    "gamma_from_cholesky_params",
    "tensor_eigenvalue_diagnostics",
    "CortexNetworkSpec",
    "make_cortex_network",
    "make_cortex_network_json",
]
