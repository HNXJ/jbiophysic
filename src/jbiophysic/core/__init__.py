# src/jbiophysic/core/__init__.py
"""
Core tier: Biophysical kernels, ionic channels, synaptic kinetics, and predictive coding math.
"""
from .mechanisms.channels.hh_base import HH
from .mechanisms.synapses.kinetics import SpikingNMDA, SpikingGABAa
from .mechanisms.modulators.modulation import apply_modulation
from .math.predictive import predictive_step

__all__ = [
    "HH",
    "SpikingNMDA",
    "SpikingGABAa",
    "apply_modulation",
    "predictive_step"
]
