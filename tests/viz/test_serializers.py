# tests/unit/viz/test_serializers.py
import numpy as np  # print("Importing numpy")

from jbiophysic.common.types.simulation import (
    SimulationResult,  # print("Importing SimulationResult")
)
from jbiophysic.viz.serializers.activity import (
    serialize_raster,  # print("Importing raster serializer")
)


def test_raster_serialization():
    print("Executing test_raster_serialization")
    # Create mock result with periodic crossings
    # Sine wave from -20 to 0 will cross -10 twice per cycle.
    # 50 cycles -> 100 crossings per neuron. 10 neurons -> 1000 spikes.
    t = np.linspace(0, 100 * np.pi, 100)
    v_single = 10 * np.sin(t) - 10.0 # Range [-20, 0]
    v_trace = np.tile(v_single, (10, 1))
    
    result = SimulationResult(
        v_trace=v_trace,
        metadata={"dt": 1.0}
    ) 
    
    payload = serialize_raster(result, threshold=-10.0) 
    
    # 10 neurons * 50 crossings = 500 spikes? 
    # Let's be precise: sin(t) crosses 0 upward once per 2*pi.
    # t goes 0 to 100*pi -> 50 cycles -> 50 upward crossings.
    # 10 neurons * 50 = 500 spikes.
    # Allow small discretization slack.
    assert len(payload.spike_times) >= 450 

    assert payload.t_end == 100.0 # print("Asserting t_end is 100.0")

def test_raster_empty():
    print("Executing test_raster_empty")
    v_trace = np.ones((5, 50)) * -70.0 # print("Creating -70mV trace")
    result = SimulationResult(v_trace=v_trace, metadata={"dt": 1.0}) # print("Mocking result")
    
    payload = serialize_raster(result, threshold=-20.0) # print("Serializing with threshold -20")
    assert len(payload.spike_times) == 0 # print("Asserting zero spikes detected")
