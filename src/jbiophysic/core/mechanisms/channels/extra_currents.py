# src/jbiophysic/core/mechanisms/channels/extra_currents.py
import jax.numpy as jnp
from jaxley.channels import Channel

class IA(Channel):
    """
    Transient A-type Potassium current (Axis 11).
    Commonly used in DynaSim for frequency-current (f-I) relationship modulation.
    """
    def __init__(self, name: str = "IA"):
        self.current_is_in_mA_per_cm2 = True
        super().__init__(name=name)
        self.channel_params = {"gka": 0.02, "ek": -90.0}
        self.channel_states = {"m": 0.0, "h": 0.8}

    def update_states(self, states, dt, v, params):
        # Activation (m)
        m_inf = 1.0 / (1.0 + jnp.exp(-(v + 60.0) / 8.5))
        tau_m = 0.37 + 1.0 / (jnp.exp((v + 35.8) / 19.7) + jnp.exp(-(v + 79.7) / 12.7))
        
        # Inactivation (h)
        h_inf = 1.0 / (1.0 + jnp.exp((v + 78.0) / 6.0))
        tau_h = 19.0 + 1.0 / (jnp.exp((v + 46.0) / 5.0) + jnp.exp(-(v + 238.0) / 37.5))
        
        # Rush-Larsen integration
        new_m = m_inf + (states["m"] - m_inf) * jnp.exp(-dt / tau_m)
        new_h = h_inf + (states["h"] - h_inf) * jnp.exp(-dt / tau_h)
        
        return {"m": jnp.clip(new_m, 0, 1), "h": jnp.clip(new_h, 0, 1)}

    def compute_current(self, states, v, params):
        return params["gka"] * (states["m"]**4) * states["h"] * (v - params["ek"])

class Ih(Channel):
    """
    Hyperpolarization-activated HCN current.
    DynaSim standard for resonance and oscillatory dynamics.
    """
    def __init__(self, name: str = "Ih"):
        self.current_is_in_mA_per_cm2 = True
        super().__init__(name=name)
        self.channel_params = {"gh": 0.0001, "eh": -43.0}
        self.channel_states = {"m": 0.1}

    def update_states(self, states, dt, v, params):
        m_inf = 1.0 / (1.0 + jnp.exp((v + 75.0) / 5.5))
        # Voltage-dependent time constant (simplified sigmoid-sum form)
        tau_m = 1.0 / (jnp.exp(-14.59 - 0.086 * v) + jnp.exp(-1.87 + 0.07 * v))
        
        new_m = m_inf + (states["m"] - m_inf) * jnp.exp(-dt / tau_m)
        return {"m": jnp.clip(new_m, 0, 1)}

    def compute_current(self, states, v, params):
        return params["gh"] * states["m"] * (v - params["eh"])

class IM(Channel):
    """
    Slow Muscarinic M-type Potassium current.
    DynaSim standard for spike-frequency adaptation.
    """
    def __init__(self, name: str = "IM"):
        self.current_is_in_mA_per_cm2 = True
        super().__init__(name=name)
        self.channel_params = {"gkm": 0.0005, "ek": -90.0, "tau_max": 1000.0}
        self.channel_states = {"m": 0.01}

    def update_states(self, states, dt, v, params):
        m_inf = 1.0 / (1.0 + jnp.exp(-(v + 35.0) / 10.0))
        tau_m = params["tau_max"] / (3.3 * jnp.exp((v + 35.0) / 20.0) + jnp.exp(-(v + 35.0) / 20.0))
        
        new_m = m_inf + (states["m"] - m_inf) * jnp.exp(-dt / tau_m)
        return {"m": jnp.clip(new_m, 0, 1)}

    def compute_current(self, states, v, params):
        return params["gkm"] * states["m"] * (v - params["ek"])
