# codes/hierarchy.py
import jaxley as jx
import jax.numpy as jnp
from codes.neurons import build_neuron
from codes.synapses import SpikingNMDA, SpikingGABAa

def build_11_area_hierarchy(alpha_pv=1.0, alpha_sst=1.0, alpha_vip=1.0):
    """
    Axis 6 + Axis 13: Multi-area hierarchy with inhibitory scaling.
    Areas: V1, V2, V4, MT, ... PFC
    """
    areas = []
    for i in range(11):
        # Create a canonical column (simplified for Axis 13)
        column = jx.Network()
        
        # Populations
        pc = build_neuron("PC")
        pv = build_neuron("PV")
        sst = build_neuron("SST")
        vip = build_neuron("VIP")
        
        # Internal Connectivity with scaling (Axis 13)
        # alpha_pv scales PV -> PC (perisomatic)
        jx.connect(pv, pc, SpikingGABAa, g=0.2 * alpha_pv)
        
        # alpha_sst scales SST -> PC (dendritic)
        jx.connect(sst, pc, SpikingGABAa, g=0.1 * alpha_sst)
        
        # alpha_vip scales VIP -> SST (disinhibition)
        jx.connect(vip, sst, SpikingGABAa, g=0.1 * alpha_vip)
        
        areas.append(column)
        
    return areas
