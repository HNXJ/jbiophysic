import jaxley as jx
import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Tuple
from modules.cortical import (
    build_pyramidal_cell, build_pv_cell, build_sst_cell, 
    build_vip_cell, build_cb_cell, build_cr_cell
)
from synapses.graded import graded_ampa, graded_gabaa, graded_gabab, graded_nmda

@dataclass
class V1PopIndices:
    pc: List[int]     # Pyramidal Cells
    pv: List[int]     # Parvalbumin
    sst: List[int]    # Somatostatin
    vip: List[int]    # Vasoactive Intestinal Peptide
    cb: List[int]     # Calbindin
    cr: List[int]     # Calretinin
    all: List[int]

def create_v1_cells():
    """Create 250 neurons for a biologically grounded microcircuit."""
    # Principal Cells (100)
    pc = [build_pyramidal_cell() for _ in range(100)]
    
    # Interneurons (150)
    pv  = [build_pv_cell() for _ in range(40)]
    sst = [build_sst_cell() for _ in range(40)]
    vip = [build_vip_cell() for _ in range(30)]
    cb  = [build_cb_cell() for _ in range(20)]
    cr  = [build_cr_cell() for _ in range(20)]
    
    cells = pc + pv + sst + vip + cb + cr
    
    indices = V1PopIndices(
        pc=list(range(0, 100)),
        pv=list(range(100, 140)),
        sst=list(range(140, 180)),
        vip=list(range(180, 210)),
        cb=list(range(210, 230)),
        cr=list(range(230, 250)),
        all=list(range(250))
    )
    return cells, indices

def wire_v1_column(net: jx.Network, pops: V1PopIndices):
    """
    Implement biologically grounded canonical microcircuit connectivity.
    Based on predictive coding and meta-disinhibitory motifs.
    """
    
    # 1. PC recurrent outputs (AMPA/NMDA)
    jx.connect(net.cell(pops.pc), net.cell(pops.pc), graded_ampa(pre=None, post=None, g=1.5), p=0.15)
    jx.connect(net.cell(pops.pc), net.cell(pops.pc), graded_nmda(pre=None, post=None, g=0.5, Mg=1.0), p=0.1)
    
    # 2. PC driving inhibition
    jx.connect(net.cell(pops.pc), net.cell(pops.pv), graded_ampa(pre=None, post=None, g=2.0), p=0.3)
    jx.connect(net.cell(pops.pc), net.cell(pops.sst), graded_ampa(pre=None, post=None, g=1.0), p=0.2) # Facilitating proxy
    
    # 3. Interneurons -> PC
    # PV -> PC soma (fast, strong)
    jx.connect(net.cell(pops.pv), net.cell(pops.pc), graded_gabaa(pre=None, post=None, g=4.0), p=0.4)
    # SST -> PC dendrite (slow, modulatory)
    jx.connect(net.cell(pops.sst), net.cell(pops.pc), graded_gabab(pre=None, post=None, g=2.5), p=0.3)
    # CB -> PC dendrite (milder)
    jx.connect(net.cell(pops.cb), net.cell(pops.pc), graded_gabaa(pre=None, post=None, g=1.5), p=0.2)
    
    # 4. Interneuron cross-modulation
    # VIP -> SST (Disinhibition)
    jx.connect(net.cell(pops.vip), net.cell(pops.sst), graded_gabaa(pre=None, post=None, g=3.0), p=0.5)
    # CR -> Interneurons (Meta-inhibition)
    jx.connect(net.cell(pops.cr), net.cell(pops.vip), graded_gabaa(pre=None, post=None, g=2.0), p=0.3)
    jx.connect(net.cell(pops.cr), net.cell(pops.cb), graded_gabaa(pre=None, post=None, g=2.0), p=0.3)

def build_v1_column(seed: int = 42):
    np.random.seed(seed)
    cells, pops = create_v1_cells()
    net = jx.Network(cells)
    wire_v1_column(net, pops)
    return net, pops

def generate_sensory_input(t_max: float, dt: float, freq: float = 40.0, amp: float = 0.2):
    t = np.arange(0, t_max, dt)
    return amp * (1.0 + np.sin(2 * np.pi * freq * t / 1000.0)) / 2.0
