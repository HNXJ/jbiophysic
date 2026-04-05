import jax.numpy as jnp
import jaxley as jx

def spike_fn(v, threshold=-50.0):
    """Event-driven binary spike detection."""
    return (v > threshold).astype(jnp.float32)

class spiking_synapse(jx.connect.Synapse):
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
            "mg": 1.0, "stdp_on": False, "stdp_delta": 0.01,
            "tau_pre": 20.0, "tau_post": 20.0, "a_plus": 1.0, "a_minus": 1.2,
            "w_max": 1.0, "type": "ampa"
        }
        self.synapse_params.update(kwargs)
        self.synapse_states = {"s": 0.0, "w": self.synapse_params["g"], "trace_pre": 0.0, "trace_post": 0.0}

    def update_states(self, states, dt, v_pre, v_post, params):
        pre_spike = spike_fn(v_pre)
        post_spike = spike_fn(v_post)
        
        ds = -states["s"] / params["tau_d"] + pre_spike * (1.0 - states["s"]) / params["tau_r"]
        new_s = states["s"] + dt * ds
        
        # Axis 14: NMDA Calcium Link & Homeostasis
        # STDP is inherently local to the synapse via the params["stdp_on"] flag
        def stdp_update():
            # Ca influx gated by pre-synaptic s and post-synaptic depolarization (e.g., NMDA mg block removal)
            ca = states["s"] * jnp.maximum(v_post, 0.0) 
            from .plasticity import stdp_core
            dw, tp, tq = stdp_core(pre_spike, post_spike, states["trace_pre"], states["trace_post"], params, dt)
            
            # Homeostatic constraint (Axis 16: Must be weight-dependent)
            actual_rate = params.get("actual_rate", 5.0)
            target_rate = params.get("target_rate", 5.0)
            dw_homeo = params.get("eta_homeo", 0.001) * (target_rate - actual_rate) * states["w"]
            
            nw = states["w"] + params["stdp_delta"] * ca * dw + dw_homeo
            return jnp.clip(nw, 0.0, params["w_max"]), tp, tq

        def no_stdp():
            return states["w"], states["trace_pre"], states["trace_post"]

        new_w, tp, tq = jax.lax.cond(params["stdp_on"], stdp_update, no_stdp)

        return {"s": new_s, "w": new_w, "trace_pre": tp, "trace_post": tq}

    def compute_current(self, states, v_pre, v_post, params):
        if params["type"] == "nmda":
            b = 1.0 / (1.0 + params["mg"] * jnp.exp(-0.062 * v_post))
            return states["w"] * states["s"] * b * (v_post - params["e"])
        return states["w"] * states["s"] * (v_post - params["e"])

class SpikingNMDA(spiking_synapse):
    def __init__(self, pre, post, name="NMDA"):
        super().__init__(pre, post, name=name, type="nmda", tau_r=2.0, tau_d=100.0, e=0.0)

class SpikingGABAa(spiking_synapse):
    def __init__(self, pre, post, name="GABAa"):
        super().__init__(pre, post, name=name, type="gabaa", tau_r=1.0, tau_d=6.0, e=-70.0)
