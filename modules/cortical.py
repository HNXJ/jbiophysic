import jaxley as jx
import jax.numpy as jnp
from channels.hh import SafeHH
from channels.neuromodulators import Dopamine, ACh

class Adaptation(jx.channels.Channel):
    """
    Slow adaptation current mechanism (e.g. M-current proxy)
    """
    def __init__(self, name="Adapt"):
        super().__init__(name=name)
        self.channel_params = {"gAdapt": 1.0, "tau_adapt": 100.0, "alpha": 0.01}
        self.channel_states = {"adapt": 0.0}

    def update_states(self, states, dt, v, params):
        d_adapt = (params["alpha"] * v - states["adapt"]) / params["tau_adapt"]
        return {"adapt": states["adapt"] + dt * d_adapt}

    def compute_current(self, states, v, params):
        return params["gAdapt"] * states["adapt"]

def build_pyramidal_cell():
    """PC: Excitatory, Moderate Adaptation, Modulated."""
    cell = jx.Cell()
    cell.insert(SafeHH())
    cell.insert(Adaptation())
    cell.insert(Dopamine()) # Axis 2
    cell.insert(ACh())      # Axis 2
    
    cell.set("HH_gNa", 120.0)
    cell.set("HH_gK", 36.0)
    cell.set("HH_gleak", 0.3)
    cell.set("Adapt_gAdapt", 1.0)
    return cell

def stdp_params_pc():
    """Axis 3: PC-PC Hebbian Plasticity."""
    return {
        "tau_pre": 20.0,
        "tau_post": 20.0,
        "a_plus": 1.0,
        "a_minus": 1.2,
        "stdp_on": True,
        "stdp_delta": 1.0
    }

def build_pv_cell():
    """PV: Fast-spiking, No Adaptation, Stabilizing."""
    cell = jx.Cell()
    cell.insert(SafeHH())
    cell.insert(Dopamine())
    cell.set("HH_gNa", 100.0)
    cell.set("HH_gK", 80.0)
    return cell

def stdp_params_pv():
    """Axis 3: Inhibitory Homeostatic Plasticity."""
    return {
        "tau_pre": 10.0,
        "tau_post": 10.0,
        "a_plus": -0.5,
        "a_minus": 0.5,
        "stdp_on": True,
        "stdp_delta": 0.2
    }

def build_sst_cell():
    """SST: Dendritic targeting, Modulated."""
    cell = jx.Cell()
    cell.insert(SafeHH())
    cell.insert(Adaptation())
    cell.insert(ACh())
    cell.set("HH_gNa", 80.0)
    cell.set("HH_gK", 30.0)
    return cell

def stdp_params_sst():
    """Axis 3: Slow, Weak STDP."""
    return {
        "tau_pre": 40.0,
        "tau_post": 40.0,
        "a_plus": 0.3,
        "a_minus": 0.4,
        "stdp_on": True,
        "stdp_delta": 0.05
    }

def build_vip_cell():
    """Disinhibitory interneuron."""
    cell = jx.Cell()
    cell.insert(SafeHH())
    cell.insert(Adaptation())
    cell.set("HH_gNa", 90.0)
    cell.set("HH_gK", 40.0)
    cell.set("HH_gleak", 0.2)
    cell.set("HH_eNa", 50.0)
    cell.set("HH_eK", -80.0)
    cell.set("HH_eleak", -60.0)
    cell.set("Adapt_gAdapt", 0.1)
    return cell

def build_cb_cell():
    """Calbindin interneuron (dendritic targeting like SST but milder)."""
    cell = jx.Cell()
    cell.insert(SafeHH())
    cell.insert(Adaptation())
    cell.set("HH_gNa", 85.0)
    cell.set("HH_gK", 35.0)
    cell.set("HH_gleak", 0.2)
    cell.set("HH_eNa", 50.0)
    cell.set("HH_eK", -80.0)
    cell.set("HH_eleak", -60.0)
    cell.set("Adapt_gAdapt", 0.3)
    return cell

def build_cr_cell():
    """Calretinin interneuron (interneuron targeting like VIP)."""
    cell = jx.Cell()
    cell.insert(SafeHH())
    cell.insert(Adaptation())
    cell.set("HH_gNa", 95.0)
    cell.set("HH_gK", 30.0)
    cell.set("HH_gleak", 0.15)
    cell.set("HH_eNa", 50.0)
    cell.set("HH_eK", -80.0)
    cell.set("HH_eleak", -62.0)
    cell.set("Adapt_gAdapt", 0.1)
    return cell
