"""Model builders and simulation entry points.

Heavy Jaxley-backed builders are optional at import time.  This keeps lightweight TFNE,
Izhikevich, and network-specification utilities usable in environments that have JAX but not
Jaxley installed.
"""

from __future__ import annotations

__all__ = ["build_cortical_hierarchy", "run_simulation"]


def build_cortical_hierarchy(*args, **kwargs):
    """Lazy wrapper around the Jaxley-dependent cortical hierarchy builder."""
    try:
        from jbiophysic.models.builders.hierarchy import build_cortical_hierarchy as _impl
    except ModuleNotFoundError as exc:
        if exc.name == "jaxley":
            raise ImportError(
                "build_cortical_hierarchy requires optional dependency 'jaxley'. "
                "Install the full jbiophysic dependencies before using this builder."
            ) from exc
        raise
    return _impl(*args, **kwargs)


def run_simulation(*args, **kwargs):
    """Lazy wrapper around the legacy simulation runner."""
    from jbiophysic.models.simulation.run import run_simulation as _impl

    return _impl(*args, **kwargs)
