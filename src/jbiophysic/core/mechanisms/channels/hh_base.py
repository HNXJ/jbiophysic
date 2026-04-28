# src/jbiophysic/core/mechanisms/channels/hh_base.py
import jax.numpy as jnp
import jaxley as jx
from jaxley.channels import HH as JaxleyHH

class HH(JaxleyHH):
    """
    Standard Hodgkin-Huxley (1952) kinetics.
    Implemented with Rush-Larsen integration and L'Hôpital-safe rate functions.
    Inherits from jaxley.channels.HH to maintain compatibility with Jaxley 0.13.0
    internal scan sorting requirements.
    """
    def __init__(self, name: str = "HH"):
        self.current_is_in_mA_per_cm2 = True
        # MANDATORY: Must use the built-in HH initialization to satisfy Jaxley's internal scan sorting
        super().__init__(name=name)
        
        # Override the defaults with the jbiophysic specific constants
        self.channel_params = {
            f"{name}_gNa": 0.12, f"{name}_gK": 0.036, f"{name}_gLeak": 0.0003,
            f"{name}_eNa": 50.0, f"{name}_eK": -77.0, f"{name}_eLeak": -54.3
        }
        self.channel_states = {f"{name}_m": 0.05, f"{name}_h": 0.6, f"{name}_n": 0.32}

    def update_states(self, states, dt, v, params):
        name = self._name
        # Helper for L'Hôpital safe rate evaluation
        def safe_rate(v_offset, scale, divisor_scale):
            # Singularity handling: jnp.where evaluates both branches, so safe_v prevents NaNs.
            safe_v = jnp.where(jnp.abs(v_offset) < 1e-6, 1.0, v_offset)
            val = scale * v_offset / (1.0 - jnp.exp(-safe_v / divisor_scale))
            return jnp.where(jnp.abs(v_offset) < 1e-6, scale * divisor_scale, val)

        # alpha_m singularity at v = -40.0
        alpha_m = safe_rate(v + 40.0, 0.1, 10.0)
        beta_m = 4.0 * jnp.exp(-(v + 65.0) / 18.0)
        
        # alpha_h and beta_h
        alpha_h = 0.07 * jnp.exp(-(v + 65.0) / 20.0)
        beta_h = 1.0 / (1.0 + jnp.exp(-(v + 35.0) / 10.0))
        
        # alpha_n singularity at v = -55.0
        alpha_n = safe_rate(v + 55.0, 0.01, 10.0)
        beta_n = 0.125 * jnp.exp(-(v + 65.0) / 80.0)

        # Rush-Larsen integration: x(t+dt) = inf + (x(t) - inf) * exp(-dt/tau)
        def rl_step(x, alpha, beta):
            tau = 1.0 / (alpha + beta)
            inf = alpha * tau
            new_x = inf + (x - inf) * jnp.exp(-dt / tau)
            return jnp.clip(new_x, 0.0, 1.0) # Physiological clamping

        new_m = rl_step(states[f"{name}_m"], alpha_m, beta_m)
        new_h = rl_step(states[f"{name}_h"], alpha_h, beta_h)
        new_n = rl_step(states[f"{name}_n"], alpha_n, beta_n)

        return {f"{name}_m": new_m, f"{name}_h": new_h, f"{name}_n": new_n}

    def compute_current(self, states, v, params):
        name = self._name
        # S/cm2 * mV = mA/cm2. No division by 1000 needed.
        ina = params[f"{name}_gNa"] * (states[f"{name}_m"]**3) * states[f"{name}_h"] * (v - params[f"{name}_eNa"])
        ik = params[f"{name}_gK"] * (states[f"{name}_n"]**4) * (v - params[f"{name}_eK"])
        il = params[f"{name}_gLeak"] * (v - params[f"{name}_eLeak"])
        return ina + ik + il