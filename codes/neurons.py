# codes/neurons.py
import jaxley as jx
import jax.numpy as jnp
from jbiophysics.channels.hh import SafeHH

def build_neuron(name="PC"):
    cell = jx.Cell()
    cell.insert(SafeHH())
    # Add other channels...
    return cell

# --- PROXY: codes/synapses.py ---
def nmda_voltage_block(v, mg=1.0):
    return 1.0 / (1.0 + mg * jnp.exp(-0.062 * v))

def nmda_step(s, spike, dt, tau_r=2.0, tau_d=100.0):
    ds = -s / tau_d + spike * (1.0 - s) / tau_r
    return s + dt * ds

def nmda_current(g, s, v, e=0.0, mg=1.0):
    b = nmda_voltage_block(v, mg)
    return g * s * b * (v - e)

# --- PROXY: codes/plasticity.py ---
def stdp_core(pre, post, trace_pre, trace_post, params, dt):
    trace_pre = trace_pre + dt * (-trace_pre / params["tau_pre"] + pre)
    trace_post = trace_post + dt * (-trace_post / params["tau_post"] + post)
    dw = params["a_plus"] * pre * trace_post - params["a_minus"] * post * trace_pre
    return dw, trace_pre, trace_post

# --- PROXY: codes/modulation.py ---
def compute_modulation(state):
    da = state.get("da", 0.0)
    ach = state.get("ach", 0.0)
    return {
        "precision": 1.0 + da,
        "nmda_gain": 1.0 + 0.5 * da,
        "stdp_scale": 1.0 + da,
        "input_gain": 1.0 + ach,
        "topdown_gain": 1.0 - 0.5 * ach
    }
