import jax
import jax.numpy as jnp
import jaxley as jx
import optax

class GradedGABAa(jx.synapses.Synapse):
    def __init__(self, g: float = 5.0, tauD_GABAa: Optional[float] = None):
        super().__init__()
        self.synapse_params = {"gGABAa": g, "EGABAa": -80.0, "tauDGABAa": 5.0, "tauRGABAa": 0.5, "slopeGABAa": 5.0, "V_thGABAa": -20.0}
        if tauD_GABAa is not None: self.synapse_params["tauDGABAa"] = tauD_GABAa
        self.synapse_states = {"sGABAa": 0.1}
    def update_states(self, states, dt, pre_v, post_v, params):
        s = states["sGABAa"]
        activation = 0.5 * (1 + jnp.tanh((pre_v - params["V_thGABAa"]) / params["slopeGABAa"]))
        d_s = (-s / params["tauDGABAa"]) + activation * ((1 - s) / params["tauRGABAa"])
        new_s = s + d_s * dt
        new_s = jnp.where(jnp.isnan(new_s) | jnp.isinf(new_s), s, new_s)
        return {"sGABAa": new_s}
    def compute_current(self, states, pre_v, post_v, params): return params["gGABAa"] * states["sGABAa"] * (post_v - params["EGABAa"])