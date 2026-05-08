import jax
import jax.numpy as jnp

from jbiophysic.simulation.edge_backend import (
    EdgeList,
    IzhikevichState,
    simulate_izhikevich_edge_jax,
)


def test_edge_list_simulation_smoke():
    n_neurons = 10
    steps = 100
    dt_ms = 0.5
    
    # All-to-all connectivity
    pre, post = jnp.meshgrid(jnp.arange(n_neurons), jnp.arange(n_neurons))
    pre = pre.flatten()
    post = post.flatten()
    # Filter out self-connections
    mask = pre != post
    pre = pre[mask]
    post = post[mask]
    
    edges = EdgeList(
        pre=pre,
        post=post,
        weight=jnp.full(pre.shape, 0.05),
        receptor_index=jnp.zeros(pre.shape, dtype=jnp.int8), # All AMPA
        delay_steps=jnp.zeros(pre.shape, dtype=jnp.int32),
        plasticity_scale=jnp.ones(pre.shape),
    )
    
    params = (
        jnp.full((n_neurons,), 0.02),
        jnp.full((n_neurons,), 0.2),
        jnp.full((n_neurons,), -65.0),
        jnp.full((n_neurons,), 8.0),
    )
    
    state0 = IzhikevichState(
        v=jnp.full((n_neurons,), -65.0),
        u=jnp.full((n_neurons,), 0.2 * -65.0),
        syn_state=jnp.zeros(pre.shape),
        weights=edges.weight,
    )
    
    drives = jnp.zeros((steps, n_neurons))
    # Inject drive into neuron 0
    drives = drives.at[:20, 0].set(15.0)
    
    final_state, (V, spikes) = simulate_izhikevich_edge_jax(
        params, state0, edges, drives, dt_ms
    )
    
    assert V.shape == (steps, n_neurons)
    assert spikes.shape == (steps, n_neurons)
    assert jnp.isfinite(V).all()
    assert jnp.isfinite(final_state.weights).all()

def test_jit_compilation():
    n_neurons = 5
    steps = 10
    dt_ms = 0.5
    
    pre = jnp.array([0, 1])
    post = jnp.array([1, 0])
    edges = EdgeList(
        pre=pre,
        post=post,
        weight=jnp.array([0.1, 0.1]),
        receptor_index=jnp.array([0, 0], dtype=jnp.int8),
        delay_steps=jnp.array([0, 0], dtype=jnp.int32),
        plasticity_scale=jnp.array([1.0, 1.0]),
    )
    
    params = (
        jnp.full((n_neurons,), 0.02),
        jnp.full((n_neurons,), 0.2),
        jnp.full((n_neurons,), -65.0),
        jnp.full((n_neurons,), 8.0),
    )
    
    state0 = IzhikevichState(
        v=jnp.full((n_neurons,), -65.0),
        u=jnp.full((n_neurons,), 0.2 * -65.0),
        syn_state=jnp.zeros(2),
        weights=edges.weight,
    )
    
    drives = jnp.zeros((steps, n_neurons))
    
    # Test that it can be JITted
    jitted_sim = jax.jit(simulate_izhikevich_edge_jax, static_argnames=("dt_ms", "plasticity_enabled", "plasticity_lr", "weight_max"))
    
    _fs, (V, _S) = jitted_sim(params, state0, edges, drives, dt_ms=dt_ms)
    assert V.shape == (steps, n_neurons)
