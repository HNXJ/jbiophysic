# tests/core/test_mechanisms.py
import jax.numpy as jnp
import pytest
from jbiophysic.core.mechanisms.channels.hh_base import HH
from jbiophysic.core.mechanisms.synapses.kinetics import SpikingNMDA

def test_hh_stability():
    """Validates HH gating variable bounds and current finiteness."""
    hh = HH()
    v = jnp.array([-65.0, -40.0, 0.0])
    dt = 0.025
    states = hh.channel_states
    params = hh.channel_params
    
    new_states = hh.update_states(states, dt, v, params)
    
    for gate in ["m", "h", "n"]:
        assert jnp.all(new_states[gate] >= 0.0)
        assert jnp.all(new_states[gate] <= 1.0)
    
    current = hh.compute_current(new_states, v, params)
    assert jnp.all(jnp.isfinite(current))

def test_nmda_gating():
    """Validates NMDA Mg-block gating and current sign."""
    nmda = SpikingNMDA(None, None)
    v_pre = jnp.array([-65.0, 0.0])
    v_post = jnp.array([-65.0, -20.0])
    dt = 0.025
    states = nmda.synapse_states
    params = nmda.synapse_params
    
    new_states = nmda.update_states(states, dt, v_pre, v_post, params)
    assert jnp.all(new_states["s"] >= 0.0)
    
    current = nmda.compute_current(new_states, v_pre, v_post, params)
    # Current is I = g * s * b(V) * (V - E). E=0.
    # At -65mV, (V-E) is negative. Current should be negative or zero.
    assert current[0] <= 0.0

def test_hh_singularities():
    """Exact-point checks for HH rate singularities at -40mV and -55mV."""
    hh = HH()
    dt = 0.025
    params = hh.channel_params
    states = hh.channel_states
    
    # Check -40.0 and -55.0 exactly
    v = jnp.array([-40.0, -55.0])
    new_states = hh.update_states(states, dt, v, params)
    
    for gate in ["m", "h", "n"]:
        assert jnp.all(jnp.isfinite(new_states[gate]))
        assert jnp.all(new_states[gate] >= 0.0)
        assert jnp.all(new_states[gate] <= 1.0)

def test_synapse_vector_broadcasting():
    """Regression test for synapse updates with vector inputs and scalar states."""
    nmda = SpikingNMDA(None, None)
    dt = 0.025
    # States start as scalars
    states = {"s": 0.0, "w": 0.1, "trace_pre": 0.0, "trace_post": 0.0}
    params = nmda.synapse_params
    
    # Vector inputs
    v_pre = jnp.array([-65.0, 0.0, -65.0])
    v_post = jnp.array([-65.0, -65.0, 0.0])
    
    # This should not raise shape errors or NaNs
    new_states = nmda.update_states(states, dt, v_pre, v_post, params)
    
    assert new_states["w"].shape == v_pre.shape
    assert jnp.all(jnp.isfinite(new_states["w"]))
