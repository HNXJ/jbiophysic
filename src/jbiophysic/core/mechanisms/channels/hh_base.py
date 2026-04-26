# src/jbiophysic/core/mechanisms/channels/hh_base.py
import jax.numpy as jnp
import jaxley as jx
from jaxley.channels import Channel

class HH(Channel):
    """
    Standard Hodgkin-Huxley (1952) kinetics.
    Implemented as a pure Jaxley channel mechanism.
    """
    def __init__(self, name: str = "HH"):
        self.current_is_in_mA_per_cm2 = True
        super().__init__(name=name)
        self.channel_params = {
            "gna": 0.12, "gk": 0.036, "gl": 0.0003,
            "ena": 50.0, "ek": -77.0, "el": -54.3
        }
        self.channel_states = {"m": 0.05, "h": 0.6, "n": 0.32}

    def update_states(self, states, dt, v, params):
        alpha_m = 0.1 * (v + 40.0) / (1.0 - jnp.exp(-(v + 40.0) / 10.0))
        beta_m = 4.0 * jnp.exp(-(v + 65.0) / 18.0)
        dm = alpha_m * (1.0 - states["m"]) - beta_m * states["m"]
        new_m = states["m"] + dt * dm

        alpha_h = 0.07 * jnp.exp(-(v + 65.0) / 20.0)
        beta_h = 1.0 / (1.0 + jnp.exp(-(v + 35.0) / 10.0))
        dh = alpha_h * (1.0 - states["h"]) - beta_h * states["h"]
        new_h = states["h"] + dt * dh

        alpha_n = 0.01 * (v + 55.0) / (1.0 - jnp.exp(-(v + 55.0) / 10.0))
        beta_n = 0.125 * jnp.exp(-(v + 65.0) / 80.0)
        dn = alpha_n * (1.0 - states["n"]) - beta_n * states["n"]
        new_n = states["n"] + dt * dn

        return {"m": new_m, "h": new_h, "n": new_n}

    def compute_current(self, states, v, params):
        ina = params["gna"] * (states["m"]**3) * states["h"] * (v - params["ena"])
        ik = params["gk"] * (states["n"]**4) * (v - params["ek"])
        il = params["gl"] * (v - params["el"])
        return (ina + ik + il) / 1000.0