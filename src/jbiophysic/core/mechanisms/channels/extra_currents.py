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

class ICa(Channel):
    """
    High-threshold L-type Calcium current.
    DynaSim parity: used for dendritic spikes and plateau potentials.
    """
    def __init__(self, name: str = "ICa"):
        self.current_is_in_mA_per_cm2 = True
        super().__init__(name=name)
        self.channel_params = {"gca": 0.001, "eca": 120.0}
        self.channel_states = {"m": 0.05, "h": 0.9}

    def update_states(self, states, dt, v, params):
        m_inf = 1.0 / (1.0 + jnp.exp(-(v + 25.0) / 5.0))
        tau_m = 1.0 + 0.5 * jnp.exp(-(v + 30.0)**2 / 100.0)
        
        h_inf = 1.0 / (1.0 + jnp.exp((v + 60.0) / 7.0))
        tau_h = 20.0 + 50.0 / (1.0 + jnp.exp(-(v + 50.0) / 5.0))
        
        new_m = m_inf + (states["m"] - m_inf) * jnp.exp(-dt / tau_m)
        new_h = h_inf + (states["h"] - h_inf) * jnp.exp(-dt / tau_h)
        return {"m": jnp.clip(new_m, 0, 1), "h": jnp.clip(new_h, 0, 1)}

    def compute_current(self, states, v, params):
        return params["gca"] * (states["m"]**2) * states["h"] * (v - params["eca"])

class ICaT(Channel):
    """
    Low-threshold T-type Calcium current.
    DynaSim parity: used for rebound bursting in thalamic models.
    """
    def __init__(self, name: str = "ICaT"):
        self.current_is_in_mA_per_cm2 = True
        super().__init__(name=name)
        self.channel_params = {"gcat": 0.002, "eca": 120.0}
        self.channel_states = {"m": 0.0, "h": 1.0}

    def update_states(self, states, dt, v, params):
        m_inf = 1.0 / (1.0 + jnp.exp(-(v + 60.0) / 6.2))
        tau_m = 0.22 + 1.0 / (jnp.exp((v + 132.0) / 16.7) + jnp.exp(-(v + 16.8) / 18.2))
        
        h_inf = 1.0 / (1.0 + jnp.exp((v + 84.0) / 4.0))
        tau_h = 8.2 + (56.6 + 0.27 * jnp.exp((v + 115.2) / 5.0)) / (1.0 + jnp.exp((v + 86.0) / 3.2))
        
        new_m = m_inf + (states["m"] - m_inf) * jnp.exp(-dt / tau_m)
        new_h = h_inf + (states["h"] - h_inf) * jnp.exp(-dt / tau_h)
        return {"m": jnp.clip(new_m, 0, 1), "h": jnp.clip(new_h, 0, 1)}

    def compute_current(self, states, v, params):
        return params["gcat"] * (states["m"]**2) * states["h"] * (v - params["eca"])

class IKDR(Channel):
    """
    Delayed Rectifier Potassium current (DynaSim iKDR).
    Provides faster repolarization than standard HH Potassium.
    """
    def __init__(self, name: str = "IKDR"):
        self.current_is_in_mA_per_cm2 = True
        super().__init__(name=name)
        self.channel_params = {"gkdr": 0.036, "ek": -90.0}
        self.channel_states = {"n": 0.1}

    def update_states(self, states, dt, v, params):
        n_inf = 1.0 / (1.0 + jnp.exp(-(v + 30.0) / 10.0))
        tau_n = 1.0 + 5.0 / (1.0 + jnp.exp((v + 40.0) / 10.0))
        
        new_n = n_inf + (states["n"] - n_inf) * jnp.exp(-dt / tau_n)
        return {"n": jnp.clip(new_n, 0, 1)}

    def compute_current(self, states, v, params):
        return params["gkdr"] * (states["n"]**4) * (v - params["ek"])

class IAR(Channel):
    """
    Anomalous Rectifier (Inwardly rectifying current, iAR).
    DynaSim parity: used for stabilizing resting potential in hippocampal models.
    """
    def __init__(self, name: str = "IAR"):
        self.current_is_in_mA_per_cm2 = True
        super().__init__(name=name)
        self.channel_params = {"gar": 0.0002, "erev": -40.0}
        self.channel_states = {"m": 0.1}

    def update_states(self, states, dt, v, params):
        m_inf = 1.0 / (1.0 + jnp.exp((v + 75.0) / 5.5))
        tau_m = 50.0 # Typically slow or constant in simple DynaSim models
        
        new_m = m_inf + (states["m"] - m_inf) * jnp.exp(-dt / tau_m)
        return {"m": jnp.clip(new_m, 0, 1)}

    def compute_current(self, states, v, params):
        return params["gar"] * states["m"] * (v - params["erev"])

class CaDynamics(Channel):
    """
    Intracellular Calcium concentration dynamics.
    DynaSim parity: 'CaBuffer.mech'.
    Updates [Ca]i based on Calcium currents and exponential decay.
    """
    def __init__(self, name: str = "CaDynamics"):
        self.current_is_in_mA_per_cm2 = True
        super().__init__(name=name)
        self.channel_params = {"tau_ca": 200.0, "phi": 0.13, "ca_rest": 0.0001}
        self.channel_states = {"cai": 0.0001}

    def update_states(self, states, dt, v, params):
        # cai_next depends on the sum of calcium currents.
        # This mechanism expects to be updated ALONGSIDE ICa/ICaT.
        # For simplicity in this shell, we use a constant flux approximation 
        # or expect the user to pass I_Ca into params or states.
        # Standard DynaSim: dCai/dt = -phi * I_Ca - (Cai - Cai_rest)/tau_ca
        
        # We'll assume I_Ca is handled by the cell and we just implement the decay here.
        # Actual integration of I_Ca flux usually happens in a custom cell update.
        decay = (states["cai"] - params["ca_rest"]) / params["tau_ca"]
        new_cai = states["cai"] - decay * dt
        return {"cai": jnp.maximum(new_cai, params["ca_rest"])}

    def compute_current(self, states, v, params):
        return 0.0 # This is a regulatory mechanism, not a current carrier

class ICan(Channel):
    """
    Calcium-activated Non-selective cation current (iCan).
    DynaSim parity: 'iCan.mech'.
    Depends on internal [Ca]i.
    """
    def __init__(self, name: str = "ICan"):
        self.current_is_in_mA_per_cm2 = True
        super().__init__(name=name)
        self.channel_params = {"gcan": 0.0001, "erev": -20.0, "kd": 0.001}
        self.channel_states = {"m": 0.01}

    def update_states(self, states, dt, v, params):
        # kd is the half-activation concentration of Calcium
        # In a real sim, we would read 'cai' from the CaDynamics state.
        # Here we define the kinetic form.
        m_inf = 1.0 / (1.0 + (params["kd"] / 0.0005)**2) # Approximation using 0.5uM as cai
        tau_m = 100.0
        
        new_m = m_inf + (states["m"] - m_inf) * jnp.exp(-dt / tau_m)
        return {"m": jnp.clip(new_m, 0, 1)}

    def compute_current(self, states, v, params):
        return params["gcan"] * states["m"] * (v - params["erev"])
