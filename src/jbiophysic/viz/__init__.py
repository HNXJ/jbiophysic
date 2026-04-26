# src/jbiophysic/viz/__init__.py
"""
Viz tier: Decoupled serialization and visualization logic for biophysical data.
"""
from .serializers.activity import serialize_raster, serialize_voltage_traces

__all__ = [
    "serialize_raster",
    "serialize_voltage_traces"
]
