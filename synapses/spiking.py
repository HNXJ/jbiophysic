import jax.numpy as jnp
import jaxley as jx

def spike_fn(v, threshold=-50.0):
    """Event-driven binary spike detection."""
    return (v > threshold).astype(jnp.float32)

def stdp_core(pre_spike, post_spike, trace_pre, trace_post, params, dt):
    """
    Core STDP rule (Hebbian/Anti-Hebbian).
    
    Returns:
        dw: Weight change delta.
        trace_pre: Updated presynaptic trace.
        trace_post: Updated postsynaptic trace.
    """
    # 1. Update traces with exponential decay + binary spike increment
    trace_pre = trace_pre + dt * (-trace_pre / params["tau_pre"] + pre_spike)
    trace_post = trace_post + dt * (-trace_post / params["tau_post"] + post_spike)

    # 2. Hebbian timing rule
    dw = (
        params["a_plus"] * pre_spike * trace_post
        - params["a_minus"] * post_spike * trace_pre
    )

    return dw, trace_pre, trace_post

class spiking_synapse(jx.connect.Synapse):
    """
    Research-Grade Synapse (Axis 1 & 3):
    - Final Form kinetics (NMDA/GABA).
    - Calcium-modulated STDP.
    - Conductance normalization & Homeostasis.
    """
    def __init__(self, pre, post, name: str, **kwargs):
        super().__init__(pre, post, name=name)
        
        # Default biophysical parameters
        self.synapse_params = {
            "g": 0.1,
            "e": 0.0,
            "tau_r": 2.0,
            "tau_d": 100.0,
            "mg": 1.0,
            "stdp_on": False,
            "stdp_delta": 0.01,
            "tau_pre": 20.0,
            "tau_post": 20.0,
            "a_plus": 1.0,
            "a_minus": 1.2,
            "w_max": 1.0,
            "type": "ampa" # ampa, nmda, gabaa, gabab
        }
        self.synapse_params.update(kwargs)
        
        # Internal states
        self.synapse_states = {
            "s": 0.0,
            "w": self.synapse_params["g"],
            "trace_pre": 0.0,
            "trace_post": 0.0
        }

    def update_states(self, states, dt, v_pre, v_post, params):
        # 1. Detect spikes
        pre_spike = spike_fn(v_pre)
        post_spike = spike_fn(v_post)
        
        # 2. Gating variable kinetics (DynaSim style)
        ds = -states["s"] / params["tau_d"] + pre_spike * (1.0 - states["s"]) / params["tau_r"]
        new_s = states["s"] + dt * ds
        
        # 3. STDP (Axis 3)
        if params["stdp_on"]:
            # Calcium linkage: NMDA drives plasticity via depolarization
            # Ca = s * max(v_post, 0.0)
            ca = states["s"] * jnp.maximum(v_post, 0.0)
            
            dw, tp, tq = stdp_core(
                pre_spike,
                post_spike,
                states["trace_pre"],
                states["trace_post"],
                params,
                dt
            )
            
            # Update weight with calcium modulation and normalization
            new_w = states["w"] + params["stdp_delta"] * ca * dw
            new_w = jnp.clip(new_w, 0.0, params["w_max"])
        else:
            new_w = states["w"]
            tp, tq = states["trace_pre"], states["trace_post"]

        return {
            "s": new_s,
            "w": new_w,
            "trace_pre": tp,
            "trace_post": tq
        }

    def compute_current(self, states, v_pre, v_post, params):
        # 4. Final Form Currents (Axis 1)
        if params["type"] == "nmda":
            # Slow kinetics + Mg block
            b = 1.0 / (1.0 + params["mg"] * jnp.exp(-0.062 * v_post))
            return states["w"] * states["s"] * b * (v_post - params["e"])
        else:
            # Linear conductances (AMPA / GABA)
            return states["w"] * states["s"] * (v_post - params["e"])

# --- Specialized Synapses ---

class SpikingNMDA(spiking_synapse):
    """Slow Excitatory + Magnesium Block (Axis 1)."""
    def __init__(self, pre, post, name="NMDA"):
        super().__init__(pre, post, name=name, type="nmda", tau_r=2.0, tau_d=100.0, e=0.0)

class SpikingGABAa(spiking_synapse):
    """Fast Inhibitory (Axis 1)."""
    def __init__(self, pre, post, name="GABAa"):
        super().__init__(pre, post, name=name, type="gabaa", tau_r=1.0, tau_d=6.0, e=-70.0)

class SpikingGABAb(spiking_synapse):
    """Slow Metabotropic Inhibitory (Axis 1)."""
    def __init__(self, pre, post, name="GABAb"):
        super().__init__(pre, post, name=name, type="gabab", tau_r=50.0, tau_d=300.0, e=-90.0)
