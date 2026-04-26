# src/jbiophysic/viz/serializers/activity.py
import numpy as np
from jbiophysic.common.types.simulation import SimulationResult
from jbiophysic.common.types.visualization import RasterPayload, TimeSeriesPayload
from jbiophysic.common.utils.logging import get_logger

logger = get_logger(__name__)

def serialize_raster(result: SimulationResult, threshold: float = -20.0) -> RasterPayload:
    """Converts voltage traces to a spike raster payload."""
    logger.info(f"Serializing raster with threshold {threshold}mV")
    v = np.array(result.v_trace)
    
    # Detect upward threshold crossings: (v[t] > threshold) AND (v[t-1] <= threshold)
    # Pad with False at the start to maintain shape
    spikes_shifted = np.roll(v, 1, axis=1)
    spikes_shifted[:, 0] = threshold + 1.0 # Ensure t=0 doesn't trigger if already above
    
    spikes = (v > threshold) & (spikes_shifted <= threshold)
    neuron_indices, time_indices = np.where(spikes)
    
    meta = result.metadata or {}
    dt = meta.get("dt", 0.025)
    spike_times = (time_indices * dt).tolist()
    
    payload = RasterPayload(
        spike_times=spike_times,
        neuron_ids=neuron_indices.tolist(),
        t_start=0.0,
        t_end=v.shape[1] * dt,
        meta={"threshold": threshold}
    )
    
    return payload

def serialize_voltage_traces(result: SimulationResult, neuron_indices: list[int]) -> TimeSeriesPayload:
    """Extracts specific voltage traces for visualization."""
    logger.info(f"Serializing voltage traces for neurons {neuron_indices}")
    v = np.array(result.v_trace)
    
    meta = result.metadata or {}
    dt = meta.get("dt", 0.025)
    
    time = (np.arange(v.shape[1]) * dt).tolist()
    values = v[neuron_indices, :].tolist()
    
    payload = TimeSeriesPayload(
        time=time,
        values=values,
        labels=[f"Neuron {i}" for i in neuron_indices]
    )
    
    return payload
