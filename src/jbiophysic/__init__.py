"""jbiophysic – biophysical modeling and TFNE science-library kernels.

Core install (`pip install -e .`) provides numpy/scipy/pandas/PyYAML utilities
only.  JAX-backed modules (cells, tfne, networks, jtfne) are available after
installing the [jax] extra::

    pip install -e ".[jax]"

Importing a JAX-backed symbol without JAX installed raises ImportError with a
descriptive message rather than a bare ModuleNotFoundError.
"""

from __future__ import annotations

# Version derived from pyproject.toml (single source of truth).
try:
    from importlib.metadata import version as _get_version

    __version__ = _get_version("jbiophysic")
except Exception:
    # Fallback if package is not installed (e.g., during development before pip install -e .).
    __version__ = "1.0.1"

__all__ = [
    # JAX-backed – populated lazily below when JAX is present.
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
    "jtfne",
]

try:
    import jax  # noqa: F401 – presence check only

    # High-level notebook-facing TFNE workflow API.
    from . import jtfne
    from .cells.hh import HHParams, simulate_hh
    from .cells.izhikevich import IzhikevichParams, simulate_izhikevich
    from .networks.cortex import (
        CortexNetworkSpec,
        make_cortex_network,
        make_cortex_network_json,
    )
    from .tfne.fields import TFNEGrid, make_regular_grid
    from .tfne.sources import (
        conservation_error,
        gaussian_mollifier,
        project_sparse_currents,
    )
    from .tfne.tensors import gamma_from_cholesky_params, tensor_eigenvalue_diagnostics

except ImportError:
    # JAX not installed – package is importable but JAX symbols are absent.
    # Users will get a clear AttributeError / ImportError at the point of use.
    pass
