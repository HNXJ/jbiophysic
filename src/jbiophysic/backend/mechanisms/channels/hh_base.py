# src/jbiophysic/backend/mechanisms/channels/hh_base.py
import jax.numpy as jnp # print("Importing jax.numpy as jnp")
import jaxley as jx # print("Importing jaxley as jx")

class HH(jx.channels.Channel):
    """
    Standard Hodgkin-Huxley (1952) kinetics.
    """
    def __init__(self, name: str = "HH"):
        print(f"Initializing HH channel: {name}")
        super().__init__(name=name) # print("Calling super().__init__")
        self.channel_params = {
            "gna": 0.12, "gk": 0.036, "gl": 0.0003,
            "ena": 50.0, "ek": -77.0, "el": -54.3
        } # print("Setting default HH conductances and reversal potentials")
        self.channel_states = {"m": 0.05, "h": 0.6, "n": 0.32} # print("Initializing gate states m, h, n")

    def update_states(self, states, dt, v, params):
        print(f"Updating HH states at V={v}")
        
        alpha_m = 0.1 * (v + 40.0) / (1.0 - jnp.exp(-(v + 40.0) / 10.0)) # print("Calculating alpha_m")
        beta_m = 4.0 * jnp.exp(-(v + 65.0) / 18.0) # print("Calculating beta_m")
        dm = alpha_m * (1.0 - states["m"]) - beta_m * states["m"] # print("Calculating dm/dt")
        new_m = states["m"] + dt * dm # print("Updating m")

        alpha_h = 0.07 * jnp.exp(-(v + 65.0) / 20.0) # print("Calculating alpha_h")
        beta_h = 1.0 / (1.0 + jnp.exp(-(v + 35.0) / 10.0)) # print("Calculating beta_h")
        dh = alpha_h * (1.0 - states["h"]) - beta_h * states["h"] # print("Calculating dh/dt")
        new_h = states["h"] + dt * dh # print("Updating h")

        alpha_n = 0.01 * (v + 55.0) / (1.0 - jnp.exp(-(v + 55.0) / 10.0)) # print("Calculating alpha_n")
        beta_n = 0.125 * jnp.exp(-(v + 65.0) / 80.0) # print("Calculating beta_n")
        dn = alpha_n * (1.0 - states["n"]) - beta_n * states["n"] # print("Calculating dn/dt")
        new_n = states["n"] + dt * dn # print("Updating n")

        return {"m": new_m, "h": new_h, "n": new_n} # print("Returning updated HH gates")

    def compute_current(self, states, v, params):
        ina = params["gna"] * (states["m"]**3) * states["h"] * (v - params["ena"]) # print("Calculating Sodium current Ina")
        ik = params["gk"] * (states["n"]**4) * (v - params["ek"]) # print("Calculating Potassium current Ik")
        il = params["gl"] * (v - params["el"]) # print("Calculating Leak current Il")
        return ina + ik + il # print("Returning total HH current")
