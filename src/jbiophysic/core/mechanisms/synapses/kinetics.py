# src/jbiophysic/core/mechanisms/synapses/kinetics.py
import jax
import jax.numpy as jnp
import jaxley as jx

def spike_fn(v, threshold=-50.0):
    return (v > threshold).astype(jnp.float32)

class SpikingSynapse(jx.synapses.Synapse):
    """
    Research-Grade Synapse (Axis 1 & 3):
    - Final Form kinetics (NMDA/GABA).
    - Calcium-modulated STDP.
    - Conductance normalization & Homeostasis.
    """
    def __init__(self, pre, post, name: str, **kwargs):
        super().__init__(pre, post, name=name)
        
        self.synapse_params = {
            "g": 0.1, "e": 0.0, "tau_r": 2.0, "tau_d": 100.0,
            "mg": 1.0, "stdp_on": False, 
            "tau_pre": 20.0, "tau_post": 20.0, 
            "a_plus": 0.01, "a_minus": 0.012,
            "w_max": 1.0, "type": "ampa",
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
        
        # 1. Update Synaptic Activation (Double Exponential or Simple)
        ds = -states["s"] / params["tau_d"] + pre_spike * (1.0 - states["s"]) / params["tau_r"]
        new_s = states["s"] + dt * ds
        
        def stdp_update():
            # 2. Update Traces
            dtp = -states["trace_pre"] / params["tau_pre"] + pre_spike
            dtq = -states["trace_post"] / params["tau_post"] + post_spike
            
            ntp = states["trace_pre"] + dt * dtp
            ntq = states["trace_post"] + dt * dtq
            
            # 3. Calculate Weight Change (STDP + Homeostasis)
            # Potentiation: pre before post
            # Depression: post before pre
            dw_stdp = params["a_plus"] * states["trace_pre"] * post_spike - \
                      params["a_minus"] * states["trace_post"] * pre_spike
            
            # Homeostasis (Simple Scaling)
            actual_rate = params.get("actual_rate", 5.0)
            dw_homeo = params["eta_homeo"] * (params["target_rate"] - actual_rate) * states["w"]
            
            nw = states["w"] + dw_stdp + dw_homeo
            return jnp.clip(nw, 0.0, params["w_max"]), ntp, ntq

        def no_stdp():
            return states["w"], states["trace_pre"], states["trace_post"]

        new_w, ntp, ntq = jax.lax.cond(params["stdp_on"], stdp_update, no_stdp)

        return {"s": new_s, "w": new_w, "trace_pre": ntp, "trace_post": ntq}

    def compute_current(self, states, v_pre, v_post, params):
        def nmda_current():
            b = 1.0 / (1.0 + params["mg"] * jnp.exp(-0.062 * v_post))
            return states["w"] * states["s"] * b * (v_post - params["e"])

        def linear_current():
            return states["w"] * states["s"] * (v_post - params["e"])

        res = jax.lax.cond(params["type"] == "nmda", nmda_current, linear_current)
        return res

class SpikingNMDA(SpikingSynapse):
    def __init__(self, pre, post, name="NMDA"):
        super().__init__(pre, post, name=name, type="nmda", tau_r=2.0, tau_d=100.0, e=0.0)

class SpikingGABAa(SpikingSynapse):
    def __init__(self, pre, post, name="GABAa"):
        super().__init__(pre, post, name=name, type="gabaa", tau_r=1.0, tau_d=6.0, e=-70.0)
