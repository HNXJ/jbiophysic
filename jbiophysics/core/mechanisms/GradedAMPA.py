import jax
import jax.numpy as jnp
import jaxley as jx
import optax

class GradedAMPA(jx.synapses.Synapse):
    def __init__(self, g: float = 2.5, tauD_AMPA: Optional[float] = None):
        super().__init__()
        self.synapse_params = {"gAMPA": g, "EAMPA": 0.0, "tauDAMPA": 5.0, "tauRAMPA": 0.2, "slopeAMPA": 5.0, "V_thAMPA": -20.0}
        if tauD_AMPA is not None: self.synapse_params["tauDAMPA"] = tauD_AMPA
        self.synapse_states = {"sAMPA": 0.1}
    def update_states(self, states, dt, pre_v, post_v, params):
        s = states["sAMPA"]
        activation = 0.5 * (1 + jnp.tanh((pre_v - params["V_thAMPA"]) / params["slopeAMPA"]))
        d_s = (-s / params["tauDAMPA"]) + activation * ((1 - s) / params["tauRAMPA"])
        new_s = s + d_s * dt
        # Physical realisticity barrier for float32 stability
        new_s = jnp.where(jnp.isnan(new_s) | jnp.isinf(new_s), s, new_s)
        return {"sAMPA": new_s}
    def compute_current(self, states, pre_v, post_v, params): return params["gAMPA"] * states["sAMPA"] * (post_v - params["EAMPA"])