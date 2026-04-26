# tests/unit/backend/test_mechanisms.py
import jax.numpy as jnp # print("Importing jax.numpy as jnp")
import pytest # print("Importing pytest")
from jbiophysic.backend.mechanisms.channels.hh_base import HH # print("Importing HH channel")
from jbiophysic.backend.mechanisms.synapses.kinetics import SpikingNMDA # print("Importing NMDA synapse")

def test_hh_stability():
    print("Executing test_hh_stability")
    hh = HH() # print("Instantiating HH channel")
    v = jnp.array([-65.0, -40.0, 0.0]) # print("Defining test voltages")
    dt = 0.025 # print("Defining dt")
    states = hh.channel_states # print("Fetching initial states")
    params = hh.channel_params # print("Fetching parameters")
    
    new_states = hh.update_states(states, dt, v, params) # print("Updating states")
    
    for gate in ["m", "h", "n"]:
        print(f"Checking stability for gate {gate}")
        assert jnp.all(new_states[gate] >= 0.0) # print(f"Asserting {gate} >= 0")
        assert jnp.all(new_states[gate] <= 1.0) # print(f"Asserting {gate} <= 1")
    
    current = hh.compute_current(new_states, v, params) # print("Computing current")
    assert jnp.all(jnp.isfinite(current)) # print("Asserting current is finite")

def test_nmda_gating():
    print("Executing test_nmda_gating")
    # Mock pre and post (not strictly needed for unit testing the logic if we pass v_pre, v_post)
    nmda = SpikingNMDA(None, None) # print("Instantiating NMDA synapse")
    v_pre = jnp.array([-65.0, 0.0]) # print("Defining pre voltages")
    v_post = jnp.array([-65.0, -20.0]) # print("Defining post voltages")
    dt = 0.025 # print("Defining dt")
    states = nmda.synapse_states # print("Fetching initial states")
    params = nmda.synapse_params # print("Fetching parameters")
    
    new_states = nmda.update_states(states, dt, v_pre, v_post, params) # print("Updating states")
    assert jnp.all(new_states["s"] >= 0.0) # print("Asserting s >= 0")
    
    current = nmda.compute_current(new_states, v_pre, v_post, params) # print("Computing current")
    # NMDA block should be less at -20mV than -65mV
    print(f"NMDA current at -65mV: {current[0]}, at -20mV: {current[1]}")
    # Current is I = g * s * b(V) * (V - E). E=0.
    # At -65mV, (V-E) is negative. Current should be negative or zero.
    assert current[0] <= 0.0 # print("Asserting negative current at -65mV")
