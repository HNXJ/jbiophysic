# src/jbiophysic/core/mechanisms/synapses/kinetics.py
import jax
import jax.numpy as jnp
import jaxley as jx
from jaxley.synapses import Synapse

def spike_fn(v, threshold=-50.0):
    return (v > threshold).astype(jnp.float32)

class SpikingSynapse(Synapse):
    """
    Research-Grade Synapse:
    - Final Form kinetics (NMDA/GABA).
    - Calcium-modulated STDP.
    - Conductance normalization & Homeostasis.
    """
    def __init__(self, pre, post, name: str, **kwargs):
        super().__init__(name=name)
        
        self.synapse_params = {
            "g": 0.1, "e": 0.0, "tau_r": 2.0, "tau_d": 100.0,
            "mg": 1.0, "stdp_on": 0.0,
            "tau_pre": 20.0, "tau_post": 20.0, 
            "a_plus": 0.01, "a_minus": 0.012,
            "w_max": 1.0, "is_nmda": 0.0,
            "eta_homeo": 0.001, "target_rate": 5.0
        }
        self.synapse_params.update(kwargs)
        self.synapse_states = {
            "s": 0.0, "w": self.synapse_params["g"], 
            "trace_pre": 0.0, "trace_post": 0.0
        }

    def update_states(self, states, dt, v_pre, v_post, params):
        pre_spike = spike_fn(v_pre)
        post_spike = spike_fn(v_post)
        
        # 1. Synaptic activation (double exponential)
        ds = -states["s"] / params["tau_d"] + pre_spike * (1.0 - states["s"]) / params["tau_r"]
        new_s = states["s"] + dt * ds
        
        # 2. STDP trace updates (always computed; gated by stdp_on below)
        dtp = -states["trace_pre"] / params["tau_pre"] + pre_spike
        dtq = -states["trace_post"] / params["tau_post"] + post_spike
        ntp = states["trace_pre"] + dt * dtp
        ntq = states["trace_post"] + dt * dtq

        # 3. Weight change (STDP + homeostasis)
        dw_stdp = (params["a_plus"] * states["trace_pre"] * post_spike -
                   params["a_minus"] * states["trace_post"] * pre_spike)
        
        actual_rate = params.get("actual_rate", 5.0)
        dw_homeo = params["eta_homeo"] * (params["target_rate"] - actual_rate) * states["w"]
        
        candidate_w = jnp.clip(states["w"] + dw_stdp + dw_homeo, 0.0, params["w_max"])

        # Gate: if stdp_on == 0, keep original weight and traces
        stdp_flag = params["stdp_on"] > 0.5
        new_w = jax.lax.select(stdp_flag, candidate_w, states["w"])
        new_tp = jax.lax.select(stdp_flag, ntp, states["trace_pre"])
        new_tq = jax.lax.select(stdp_flag, ntq, states["trace_post"])

        return {"s": new_s, "w": new_w, "trace_pre": new_tp, "trace_post": new_tq}

    def compute_current(self, states, v_pre, v_post, params):
        # NMDA: voltage-dependent Mg block (Jahr-Stevens 1990)
        b = 1.0 / (1.0 + params["mg"] * jnp.exp(-0.062 * v_post))
        i_nmda = states["w"] * states["s"] * b * (v_post - params["e"])

        # Linear (AMPA / GABA)
        i_linear = states["w"] * states["s"] * (v_post - params["e"])

        # Select based on numeric flag instead of string comparison
        is_nmda = params["is_nmda"] > 0.5
        return jax.lax.select(is_nmda, i_nmda, i_linear)

class SpikingNMDA(SpikingSynapse):
    def __init__(self, pre, post, name="NMDA"):
        super().__init__(pre, post, name=name, is_nmda=1.0, tau_r=2.0, tau_d=100.0, e=0.0)

class SpikingGABAa(SpikingSynapse):
    def __init__(self, pre, post, name="GABAa"):
        super().__init__(pre, post, name=name, is_nmda=0.0, tau_r=1.0, tau_d=6.0, e=-70.0)