# codes/simulation.py
import jax
import jax.numpy as jnp
from typing import Dict, Any, Tuple

def delay_buffer(states_history: jnp.ndarray, delay_steps: int) -> jnp.ndarray:
    """Axis 14: Circular delay buffer for inter-areal communication."""
    # Assuming states_history shape is [buffer_size, features]
    # In scan, this is often handled by keeping track of the last N states.
    return states_history[-delay_steps]

def cortical_hierarchy_simulation_engine(
    init_state: Dict[str, jnp.ndarray],
    params: Dict[str, Any],
    T: int,
    dt: float = 0.1
) -> Tuple[Dict[str, jnp.ndarray], Dict[str, jnp.ndarray]]:
    """
    Axis 14: Fully vectorized JAX simulation engine.
    Replaces Python for loops with jax.lax.scan for 10-100x speedups.
    """
    
    def scan_step(carry, xs_t):
        current_state, state_history = carry
        
        # 1. Biological Delays (Critical for Gamma/Beta mismatch)
        # Fetch states from delay_steps ago
        delay_ff = params.get("delay_ff_steps", 10)
        delay_fb = params.get("delay_fb_steps", 20)
        
        l23_ff_input = delay_buffer(state_history["l23"], delay_ff)
        l5_fb_input = delay_buffer(state_history["l5"], delay_fb)
        
        # 2. Local Dynamics & Synaptic Updates (Vectorized)
        # Placeholder for Jaxley cell updates
        noise = params.get("sigma", 0.0) * jax.random.normal(jax.random.PRNGKey(0), current_state["v"].shape)
        
        dv = -current_state["v"] / params["tau_m"] + jnp.dot(params["W_ff"], l23_ff_input) + jnp.dot(params["W_fb"], l5_fb_input) + noise
        new_v = current_state["v"] + dt * dv
        
        # Update Area Signals
        new_state = {
            "v": new_v,
            "l23": jnp.tanh(new_v), # Output proxy
            "l5": jnp.maximum(0.0, new_v) # Output proxy
        }
        
        # 3. Update History Buffer
        # Roll buffer and append new state
        new_history = {
            k: jnp.roll(v, shift=-1, axis=0).at[-1].set(new_state[k])
            for k, v in state_history.items()
        }
        
        # What is returned: (carry, output)
        return (new_state, new_history), new_state

    # History buffer initialization
    max_delay = max(params.get("delay_ff_steps", 10), params.get("delay_fb_steps", 20)) + 1
    init_history = {
        k: jnp.tile(v, (max_delay,) + (1,) * v.ndim)
        for k, v in init_state.items()
    }

    # Run jax.lax.scan
    time_indices = jnp.arange(T)
    (final_state, final_history), trajectory = jax.lax.scan(
        scan_step,
        (init_state, init_history),
        time_indices
    )
    
    return final_state, trajectory
