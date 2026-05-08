import pytest
import jax.numpy as jnp
from jbiophysic.conditions import Condition
from jbiophysic.objectives import Objective
from jbiophysic.circuits import Circuit, SimulationResult
from jbiophysic import ops

def test_condition_metadata_isolation():
    c1 = Condition("stim", 100.0, 0.1, metadata={"a": 1})
    c2 = Condition("rest", 100.0, 0.1)
    assert c1.metadata["a"] == 1
    assert "a" not in c2.metadata

def test_objective_default_weight():
    obj = Objective("metric", 10.0)
    assert obj.weight == 1.0

def test_ops_firing_rate():
    # 100ms, 0.1ms dt -> 1000 steps
    # 1 spike at t=0 for neuron 0
    spikes = jnp.zeros((1000, 2))
    spikes = spikes.at[0, 0].set(1.0)
    dt_ms = 0.1
    
    # 1 spike in 0.1s = 10Hz
    rates = ops.per_neuron_firing_rate(spikes, dt_ms)
    assert jnp.allclose(rates[0], 10.0)
    assert rates[1] == 0.0
    
    rate = ops.firing_rate(spikes, dt_ms)
    assert hasattr(rate, "shape") or "jax" in type(rate).__module__.lower()
    assert jnp.allclose(rate, 5.0)
    
    max_rate = ops.max_single_neuron_rate(spikes, dt_ms)
    assert hasattr(max_rate, "shape") or "jax" in type(max_rate).__module__.lower()
    assert jnp.allclose(max_rate, 10.0)

def test_ops_fano_factor():
    counts = jnp.array([10, 12, 8, 10, 10])
    # mean = 10, var = (0+4+4+0+0)/5 = 1.6
    # Fano = 1.6 / 10 = 0.16
    f = ops.fano_factor(counts)
    assert jnp.allclose(f, 0.16)

def test_ops_lfp_proxy():
    csd = jnp.ones((10, 5))
    lfp = ops.lfp_proxy(csd)
    assert lfp.shape == (10,)
    assert jnp.all(lfp == 1.0)
