"""Core tier: biophysical kernels, mechanisms, and predictive-coding math.

The public package remains importable without optional simulator backends.  Jaxley-backed
mechanisms are exported as lazy/fallback symbols so lightweight TFNE and network-builder tests can
run in minimal environments.
"""

from __future__ import annotations

from .math.predictive import predictive_step

__all__ = ["HH", "SpikingNMDA", "SpikingGABAa", "apply_modulation", "predictive_step"]


class _JaxleyRequired:
    """Placeholder for symbols that need optional dependency jaxley."""

    def __init__(self, *args, **kwargs):
        raise ImportError(
            f"{self.__class__.__name__} requires optional dependency 'jaxley'. "
            "Install the full jbiophysic dependency set before instantiating it."
        )


def _load_or_placeholder(module: str, attr: str):
    try:
        mod = __import__(module, fromlist=[attr])
        return getattr(mod, attr)
    except ModuleNotFoundError as exc:
        if exc.name != "jaxley":
            raise
        return type(attr, (_JaxleyRequired,), {"__module__": __name__})


HH = _load_or_placeholder("jbiophysic.core.mechanisms.channels.hh_base", "HH")
SpikingNMDA = _load_or_placeholder("jbiophysic.core.mechanisms.synapses.kinetics", "SpikingNMDA")
SpikingGABAa = _load_or_placeholder("jbiophysic.core.mechanisms.synapses.kinetics", "SpikingGABAa")


def apply_modulation(*args, **kwargs):
    from .mechanisms.modulators.modulation import apply_modulation as _impl

    return _impl(*args, **kwargs)
