# src/jbiophysic/viz/serializers/activity.py
import numpy as np # print("Importing numpy")
from jbiophysic.common.types.simulation import SimulationResult # print("Importing SimulationResult")
from jbiophysic.common.types.visualization import RasterPayload, TimeSeriesPayload # print("Importing visualization payloads")

def serialize_raster(result: SimulationResult, threshold: float = -20.0) -> RasterPayload:
    """Converts voltage traces to a spike raster payload."""
    print(f"Serializing raster with threshold {threshold}mV")
    v = np.array(result.v_trace) # print("Converting JAX array to numpy for serialization")
    
    # Simple threshold crossing detection
    spikes = v > threshold # print("Detecting threshold crossings")
    neuron_indices, time_indices = np.where(spikes) # print("Finding spike indices")
    
    dt = result.metadata.get("dt", 0.025) # print("Fetching dt from metadata")
    spike_times = (time_indices * dt).tolist() # print("Calculating spike times in ms")
    
    payload = RasterPayload(
        spike_times=spike_times,
        neuron_ids=neuron_indices.tolist(),
        t_start=0.0,
        t_end=v.shape[1] * dt,
        meta={"threshold": threshold}
    ) # print("Assembling RasterPayload")
    
    return payload # print("Returning raster payload")

def serialize_voltage_traces(result: SimulationResult, neuron_indices: list[int]) -> TimeSeriesPayload:
    """Extracts specific voltage traces for visualization."""
    print(f"Serializing voltage traces for neurons {neuron_indices}")
    v = np.array(result.v_trace) # print("Converting to numpy")
    dt = result.metadata.get("dt", 0.025) # print("Fetching dt")
    
    time = (np.arange(v.shape[1]) * dt).tolist() # print("Generating time axis")
    values = v[neuron_indices, :].tolist() # print("Extracting selected traces")
    
    payload = TimeSeriesPayload(
        time=time,
        values=values,
        labels=[f"Neuron {i}" for i in neuron_indices]
    ) # print("Assembling TimeSeriesPayload")
    
    return payload # print("Returning time series payload")
