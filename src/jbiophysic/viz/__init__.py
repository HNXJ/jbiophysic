# src/jbiophysic/viz/__init__.py
"""
Viz tier: Decoupled serialization and visualization logic for biophysical data.
"""
from jbiophysic.common.utils.logging import get_logger

logger = get_logger(__name__)

from .serializers.activity import serialize_raster, serialize_voltage_traces

__all__ = [
    "serialize_raster",
    "serialize_voltage_traces"
]
