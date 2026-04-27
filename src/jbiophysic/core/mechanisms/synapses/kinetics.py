# src/jbiophysic/core/mechanisms/synapses/kinetics.py
import jax
import jax.numpy as jnp
import jaxley as jx
from jaxley.synapses import Synapse

def spike_fn(v, threshold=-50.0, k=5.0):
    """Differentiable surrogate spike function."""
    return jax.nn.sigmoid(k * (v - threshold))

class SpikingSynapse(Synapse):
    """
    Experimental synapse with multi-component kinetics and plasticity hooks:
    - Multi-exponential kinetics (AMPA/NMDA/GABA).
    - Calcium-modulated STDP hooks.
    - Basic conductance homeostasis.
    """
    def __init__(self, pre, post, name: str, **kwargs):
        super().__init__(name=name)
        
        self.synapse_params = {
            "g": 0.1, "e": 0.0, "tau_r": 2.0, "tau_d": 100.0,
            "mg": 1.0, "stdp_on": 0.0,
            "tau_pre": 20.0, "tau_post": 20.0, 
            "a_plus": 0.01, "a_minus": 0.012,
            "w_max": 1.0, "is_nmda": 0.0,
            "eta_homeo": 0.001, "target_rate": 5.0, "actual_rate": 5.0
        }
        self.synapse_params.update(kwargs)
        self.synapse_states = {
            "s": 0.0, "w": self.synapse_params["g"], 
            "trace_pre": 0.0, "trace_post": 0.0
        }

    def update_states(self, states, dt, v_pre, v_post, params):
        # Force broadcasting of scalar states to input shape for vector compatibility
        s_broad = jnp.broadcast_to(states["s"], v_pre.shape)
        w_broad = jnp.broadcast_to(states["w"], v_pre.shape)
        tp_broad = jnp.broadcast_to(states["trace_pre"], v_pre.shape)
        tq_broad = jnp.broadcast_to(states["trace_post"], v_pre.shape)
        
        pre_spike = spike_fn(v_pre)
        post_spike = spike_fn(v_post)
        
        # 1. Synaptic activation
        ds = -s_broad / params["tau_d"] + pre_spike * (1.0 - s_broad) / params["tau_r"]
        new_s = s_broad + dt * ds
        
        # 2. STDP trace updates
        dtp = -tp_broad / params["tau_pre"] + pre_spike
        dtq = -tq_broad / params["tau_post"] + post_spike
        ntp = tp_broad + dt * dtp
        ntq = tq_broad + dt * dtq
        
        # 3. Weight change
        dw_stdp = (params["a_plus"] * tp_broad * post_spike -
                   params["a_minus"] * tq_broad * pre_spike)
        
        actual_rate = params["actual_rate"]
        dw_homeo = params["eta_homeo"] * (params["target_rate"] - actual_rate) * w_broad
        candidate_w = jnp.clip(w_broad + dw_stdp + dw_homeo, 0.0, params["w_max"])
        
        # Gate: using jax.lax.select for shape-strict differentiable gating
        stdp_flag = params["stdp_on"] > 0.5
        new_w = jax.lax.select(stdp_flag, candidate_w, w_broad)
        new_tp = jax.lax.select(stdp_flag, ntp, tp_broad)
        new_tq = jax.lax.select(stdp_flag, ntq, tq_broad)
        
        return {"s": new_s, "w": new_w, "trace_pre": new_tp, "trace_post": new_tq}

    def compute_current(self, states, v_pre, v_post, params):
        b = 1.0 / (1.0 + (params["mg"] / 3.57) * jnp.exp(-0.062 * v_post))
        i_nmda = states["w"] * states["s"] * b * (v_post - params["e"])
        i_linear = states["w"] * states["s"] * (v_post - params["e"])
        
        is_nmda = params["is_nmda"] > 0.5
        return jax.lax.select(is_nmda, i_nmda, i_linear)

class SpikingNMDA(SpikingSynapse):
    def __init__(self, pre, post, name="NMDA"):
        super().__init__(pre, post, name=name, is_nmda=1.0, tau_r=2.0, tau_d=100.0, e=0.0)

class SpikingGABAa(SpikingSynapse):
    def __init__(self, pre, post, name="GABAa"):
        super().__init__(pre, post, name=name, is_nmda=0.0, tau_r=1.0, tau_d=6.0, e=-70.0)

class SpikingGABAb(SpikingSynapse):
    """
    Slow GABAb-mediated inhibition.
    DynaSim parity: 'iGABAb.mech'.
    """
    def __init__(self, pre, post, name="GABAb"):
        super().__init__(pre, post, name=name, is_nmda=0.0, tau_r=50.0, tau_d=200.0, e=-90.0)

class GapJunction(Synapse):
    """
    Electrical Gap Junction.
    DynaSim parity: 'iGJ.mech'.
    Current: I = g * (v_post - v_pre)
    """
    def __init__(self, pre, post, name="GapJunction"):
        super().__init__(name=name)
        self.synapse_params = {"g": 0.1}
        self.synapse_states = {}

    def update_states(self, states, dt, v_pre, v_post, params):
        return {}

    def compute_current(self, states, v_pre, v_post, params):
        return params["g"] * (v_post - v_pre)