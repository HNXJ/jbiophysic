import jaxley as jx
import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Tuple
from jbiophysics.neurons.cortical import build_pyramidal_cell, build_pv_cell, build_sst_cell, build_vip_cell

@dataclass
class V1PopIndices:
    l23_pyr: List[int]
    l4_pyr: List[int]
    l56_pyr: List[int]
    pv: List[int]
    sst: List[int]
    vip: List[int]
    all: List[int]

def create_v1_cells():
    """Create 200 neurons for a single V1 column."""
    cells = []
    # L2/3 Pyr (40), L4 Pyr (20), L5/6 Pyr (40)
    l23_pyr = [build_pyramidal_cell() for _ in range(40)]
    l4_pyr  = [build_pyramidal_cell() for _ in range(20)]
    l56_pyr = [build_pyramidal_cell() for _ in range(40)]
    
    # Inhibitory (40 PV, 40 SST, 20 VIP)
    pv  = [build_pv_cell() for _ in range(40)]
    sst = [build_sst_cell() for _ in range(40)]
    vip = [build_vip_cell() for _ in range(20)]
    
    cells = l23_pyr + l4_pyr + l56_pyr + pv + sst + vip
    
    indices = V1PopIndices(
        l23_pyr=list(range(0, 40)),
        l4_pyr=list(range(40, 60)),
        l56_pyr=list(range(60, 100)),
        pv=list(range(100, 140)),
        sst=list(range(140, 180)),
        vip=list(range(180, 200)),
        all=list(range(200))
    )
    return cells, indices

def wire_v1_column(net: jx.Network, pops: V1PopIndices):
    """Implement canonical microcircuit connectivity."""
    # This is a simplified version of the wiring described in the instructions
    # In a real scenario, we'd use NetBuilder, but for the 'systems' module 
    # we implement the explicit jaxley connections.
    
    # Recurrent Excitation
    jx.connect(net.cell(pops.l4_pyr), net.cell(pops.l23_pyr), jx.synapses.AMPA(g=1.5), p=0.2)
    jx.connect(net.cell(pops.l23_pyr), net.cell(pops.l56_pyr), jx.synapses.AMPA(g=1.0), p=0.1)
    
    # Inhibition
    jx.connect(net.cell(pops.pv), net.cell(pops.l4_pyr), jx.synapses.GABAa(g=4.0), p=0.4)
    jx.connect(net.cell(pops.pv), net.cell(pops.l23_pyr), jx.synapses.GABAa(g=4.0), p=0.4)
    jx.connect(net.cell(pops.sst), net.cell(pops.l23_pyr), jx.synapses.GABAb(g=2.0), p=0.3)
    jx.connect(net.cell(pops.vip), net.cell(pops.sst), jx.synapses.GABAa(g=3.0), p=0.5)

def build_v1_column(seed: int = 42):
    np.random.seed(seed)
    cells, pops = create_v1_cells()
    net = jx.Network(cells)
    wire_v1_column(net, pops)
    return net, pops

def generate_sensory_input(t_max: float, dt: float, freq: float = 40.0, amp: float = 0.2):
    t = np.arange(0, t_max, dt)
    # Sinusoidal drive at gamma frequency
    return amp * (1.0 + np.sin(2 * np.pi * freq * t / 1000.0)) / 2.0
