"""Optimizer primitives for bounded biophysical model search.

Optax-free helpers (``bounds``, ``manifests``) are always importable.
Optax-backed optimizers (AGSDR, GSDR, GSGD, SDR) require the [jax] extra::

    pip install -e ".[jax]"

Importing this package without Optax will not raise at import time; the
Optax-backed names simply will not be present in the namespace.
"""

from __future__ import annotations

# Always available – no JAX / Optax dependency.
from .bounds import Bound, positive_softplus, sigmoid_bounded
from .manifests import OptimizerManifest

__all__ = [
    "Bound",
    "positive_softplus",
    "sigmoid_bounded",
    "OptimizerManifest",
    # Optax-backed – added below when available.
    "AGSDR",
    "AGSDRSchedule",
    "adapt_alpha",
    "GSDR",
    "GSDRState",
    "gsdr_direction",
    "GSGD",
    "gsgd_step",
    "SDR",
    "SDRState",
    "supervised_delta_direction",
]

try:
    import optax  # noqa: F401 – presence check only

    from .agsdr import AGSDR, AGSDRSchedule, adapt_alpha
    from .gsdr import GSDR, GSDRState, gsdr_direction
    from .gsgd import GSGD, gsgd_step
    from .sdr import SDR, SDRState, supervised_delta_direction

except ImportError:
    # Optax not installed – Optax-backed optimizers are unavailable.
    # Core bounds helpers remain importable for parameter-space utilities.
    pass
