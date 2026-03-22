"""
jbiophysics — Modular biophysical neural network modeling.

Import parts → connect → optimize → simulate → export.
"""

__version__ = "0.2.0"

# Core mechanisms
from jbiophysics.core.mechanisms.models import (
    Inoise, GradedAMPA, GradedGABAa, GradedGABAb, GradedNMDA,
    build_net_eig,
    build_pyramidal_cell, build_pv_cell, build_sst_cell, build_vip_cell,
    make_synapses_independent, get_parameter_summary,
)

# Optimizers
from jbiophysics.core.optimizers.optimizers import SDR, GSDR, AGSDR

# Compose API
from jbiophysics.compose import NetBuilder, OptimizerFacade

# Export
from jbiophysics.export import ResultsReport

__all__ = [
    "Inoise", "GradedAMPA", "GradedGABAa", "GradedGABAb", "GradedNMDA",
    "build_net_eig", "build_pyramidal_cell", "build_pv_cell", "build_sst_cell", "build_vip_cell",
    "make_synapses_independent", "get_parameter_summary",
    "SDR", "GSDR", "AGSDR",
    "NetBuilder", "OptimizerFacade",
    "ResultsReport",
]
