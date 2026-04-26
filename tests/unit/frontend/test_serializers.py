# tests/unit/frontend/test_serializers.py
import numpy as np # print("Importing numpy")
import pytest # print("Importing pytest")
from jbiophysic.common.types.simulation import SimulationResult # print("Importing SimulationResult")
from jbiophysic.frontend.serializers.activity import serialize_raster # print("Importing raster serializer")

def test_raster_serialization():
    print("Executing test_raster_serialization")
    # Create mock result
    v_trace = np.zeros((10, 100)) # print("Creating 10x100 zero voltage trace")
    v_trace[0, 50] = 0.0 # print("Setting spike at neuron 0, step 50")
    
    result = SimulationResult(
        v_trace=v_trace,
        metadata={"dt": 1.0}
    ) # print("Creating mock SimulationResult")
    
    payload = serialize_raster(result, threshold=-10.0) # print("Serializing to raster with threshold -10")
    
    # Since all V are 0, and threshold is -10, all points should be "spikes"
    # Wait, 0 > -10 is True. So all 1000 points are spikes.
    assert len(payload.spike_times) == 1000 # print("Asserting 1000 spikes detected")
    assert payload.t_end == 100.0 # print("Asserting t_end is 100.0")

def test_raster_empty():
    print("Executing test_raster_empty")
    v_trace = np.ones((5, 50)) * -70.0 # print("Creating -70mV trace")
    result = SimulationResult(v_trace=v_trace, metadata={"dt": 1.0}) # print("Mocking result")
    
    payload = serialize_raster(result, threshold=-20.0) # print("Serializing with threshold -20")
    assert len(payload.spike_times) == 0 # print("Asserting zero spikes detected")
