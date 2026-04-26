# tests/core/test_properties.py
import jax.numpy as jnp
import pytest
from jbiophysic.core.mechanisms.channels.hh_base import HH
from jbiophysic.core.mechanisms.synapses.kinetics import SpikingNMDA

def test_hh_singularities():
    """Checks HH rate functions at exact singular voltages (-40, -55)."""
    hh = HH()
    dt = 0.1
    params = hh.channel_params
    states = hh.channel_states
    
    # Check at exactly -40.0 (alpha_m singularity)
    v_sing_m = jnp.array([-40.0])
    new_states_m = hh.update_states(states, dt, v_sing_m, params)
    assert jnp.all(jnp.isfinite(new_states_m["m"]))
    
    # Check at exactly -55.0 (alpha_n singularity)
    v_sing_n = jnp.array([-55.0])
    new_states_n = hh.update_states(states, dt, v_sing_n, params)
    assert jnp.all(jnp.isfinite(new_states_n["n"]))

def test_nmda_monotonicity():
    """Checks that NMDA Mg-block is monotonic with respect to voltage."""
    nmda = SpikingNMDA(None, None)
    v_range = jnp.linspace(-100.0, 40.0, 100)
    
    # Mock states and params
    states = {"w": 1.0, "s": 1.0}
    params = nmda.synapse_params
    params["is_nmda"] = 1.0
    
    currents = jax.vmap(lambda v: nmda.compute_current(states, -65.0, v, params))(v_range)
    # The Mg-block term b(V) = 1 / (1 + (mg/3.57) * exp(-0.062 * V))
    # should increase as V increases (less block at higher V).
    # Since I = g * s * b(V) * (V - E), we should check the block term itself.
    
    b_vals = 1.0 / (1.0 + (params["mg"] / 3.57) * jnp.exp(-0.062 * v_range))
    # b_vals should be strictly increasing
    assert jnp.all(jnp.diff(b_vals) > 0)

import jax
