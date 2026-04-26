# src/jbiophysic/core/mechanisms/channels/hh_base.py
import jax.numpy as jnp
import jaxley as jx
from jaxley.channels import Channel

class HH(Channel):
    """
    Standard Hodgkin-Huxley (1952) kinetics.
    Implemented with Rush-Larsen integration and L'Hôpital-safe rate functions.
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
        # alpha_m singularity at v = -40.0
        v_m = v + 40.0
        # Safe-division pattern for JAX: evaluate both branches but avoid NaNs in inactive ones.
        safe_v_m = jnp.where(jnp.abs(v_m) < 1e-6, 1.0, v_m)
        alpha_m = jnp.where(jnp.abs(v_m) < 1e-6, 1.0, 0.1 * v_m / (1.0 - jnp.exp(-safe_v_m / 10.0)))
        beta_m = 4.0 * jnp.exp(-(v + 65.0) / 18.0)
        
        # alpha_h and beta_h
        alpha_h = 0.07 * jnp.exp(-(v + 65.0) / 20.0)
        beta_h = 1.0 / (1.0 + jnp.exp(-(v + 35.0) / 10.0))
        
        # alpha_n singularity at v = -55.0
        v_n = v + 55.0
        safe_v_n = jnp.where(jnp.abs(v_n) < 1e-6, 1.0, v_n)
        alpha_n = jnp.where(jnp.abs(v_n) < 1e-6, 0.1, 0.01 * v_n / (1.0 - jnp.exp(-safe_v_n / 10.0)))
        beta_n = 0.125 * jnp.exp(-(v + 65.0) / 80.0)

        # Rush-Larsen integration: x(t+dt) = inf + (x(t) - inf) * exp(-dt/tau)
        def rl_step(x, alpha, beta):
            tau = 1.0 / (alpha + beta)
            inf = alpha * tau
            return inf + (x - inf) * jnp.exp(-dt / tau)

        new_m = rl_step(states["m"], alpha_m, beta_m)
        new_h = rl_step(states["h"], alpha_h, beta_h)
        new_n = rl_step(states["n"], alpha_n, beta_n)

        return {"m": new_m, "h": new_h, "n": new_n}

    def compute_current(self, states, v, params):
        # S/cm2 * mV = mA/cm2. No division by 1000 needed.
        ina = params["gna"] * (states["m"]**3) * states["h"] * (v - params["ena"])
        ik = params["gk"] * (states["n"]**4) * (v - params["ek"])
        il = params["gl"] * (v - params["el"])
        return ina + ik + il