import jaxley as jx
import jax.numpy as jnp
from channels.hh import SafeHH

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
    cell = jx.Cell()
    cell.insert(SafeHH())
    cell.insert(Adaptation())
    cell.set("HH_gNa", 120.0)
    cell.set("HH_gK", 36.0)
    cell.set("HH_gleak", 0.3)
    cell.set("HH_eNa", 50.0)
    cell.set("HH_eK", -77.0)
    cell.set("HH_eleak", -54.4)
    cell.set("Adapt_gAdapt", 1.0)
    return cell

def build_pv_cell():
    """Fast-spiking interneuron (soma-targeting)."""
    cell = jx.Cell()
    cell.insert(SafeHH())
    cell.set("HH_gNa", 100.0)
    cell.set("HH_gK", 80.0)
    cell.set("HH_gleak", 0.1)
    cell.set("HH_eNa", 50.0)
    cell.set("HH_eK", -90.0)
    cell.set("HH_eleak", -65.0)
    return cell

def build_sst_cell():
    """Dendrite-targeting interneuron, moderate adaptation."""
    cell = jx.Cell()
    cell.insert(SafeHH())
    cell.insert(Adaptation())
    cell.set("HH_gNa", 80.0)
    cell.set("HH_gK", 30.0)
    cell.set("HH_gleak", 0.2)
    cell.set("HH_eNa", 50.0)
    cell.set("HH_eK", -80.0)
    cell.set("HH_eleak", -60.0)
    cell.set("Adapt_gAdapt", 0.5)
    return cell

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
