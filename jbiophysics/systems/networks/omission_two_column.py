import jaxley as jx
import numpy as np
import jax.numpy as jnp
from dataclasses import dataclass
from typing import List, Dict, Tuple
from jbiophysics.systems.networks.omission_v1_column import create_v1_cells, wire_v1_column, V1PopIndices

@dataclass
class OmissionTrialConfig:
    t_total_ms: float = 5000.0
    dt_ms: float = 0.025
    bu_on: bool = True
    td_on: bool = True
    td_amp: float = 0.5
    stim_freq: float = 40.0

@dataclass
class OmissionNetwork:
    net: jx.Network
    v1_pops: V1PopIndices
    ho_pops: V1PopIndices
    n_v1: int
    n_ho: int

def build_omission_network(seed: int = 42):
    np.random.seed(seed)
    
    # 1. Create sub-networks
    v1_cells, v1_pops = create_v1_cells()
    ho_cells, ho_pops = create_v1_cells() # HO uses same cell set
    
    n_v1 = len(v1_cells)
    n_ho = len(ho_cells)
    
    # 2. Combine into one Network
    net = jx.Network(v1_cells + ho_cells)
    
    # 3. Wire internal columns
    # We need to shift indices for HO
    ho_pops_shifted = V1PopIndices(
        l23_pyr=[i + n_v1 for i in ho_pops.l23_pyr],
        l4_pyr=[i + n_v1 for i in ho_pops.l4_pyr],
        l56_pyr=[i + n_v1 for i in ho_pops.l56_pyr],
        pv=[i + n_v1 for i in ho_pops.pv],
        sst=[i + n_v1 for i in ho_pops.sst],
        vip=[i + n_v1 for i in ho_pops.vip],
        all=[i + n_v1 for i in ho_pops.all]
    )
    
    wire_v1_column(net, v1_pops)
    wire_v1_column(net, ho_pops_shifted)
    
    # 4. Inter-areal Feedforward (V1 -> HO)
    # V1 L2/3 Pyr -> HO L4 Pyr (AMPA)
    jx.connect(net.cell(v1_pops.l23_pyr), net.cell(ho_pops_shifted.l4_pyr), jx.synapses.AMPA(g=1.2), p=0.15)
    
    # 5. Inter-areal Feedback (HO -> V1)
    # HO L5/6 Pyr -> V1 L2/3 Pyr (AMPA + NMDA)
    jx.connect(net.cell(ho_pops_shifted.l56_pyr), net.cell(v1_pops.l23_pyr), jx.synapses.AMPA(g=0.8), p=0.1)
    # Feedback targets VIP for disinhibition too
    jx.connect(net.cell(ho_pops_shifted.l56_pyr), net.cell(v1_pops.vip), jx.synapses.AMPA(g=1.5), p=0.3)
    
    return OmissionNetwork(net, v1_pops, ho_pops_shifted, n_v1, n_ho)

def make_context_inputs(config: OmissionTrialConfig, v1_pops: V1PopIndices, ho_pops: V1PopIndices, n_v1: int):
    t_steps = int(config.t_total_ms / config.dt_ms)
    total_neurons = n_v1 + len(ho_pops.all)
    currents = np.zeros((total_neurons, t_steps))
    
    t = np.arange(0, config.t_total_ms, config.dt_ms)
    
    # Base noisy background (handled by Inoise mostly, but we can add bias here)
    
    # BU Input (to V1 L4)
    if config.bu_on:
        bu_pulse = 0.3 * (1.0 + np.sin(2 * np.pi * config.stim_freq * t / 1000.0)) / 2.0
        for idx in v1_pops.l4_pyr:
            currents[idx] += bu_pulse
            
    # TD Input (to HO populations)
    if config.td_on:
        td_bias = config.td_amp * np.ones_like(t)
        for idx in ho_pops.l23_pyr + ho_pops.l56_pyr:
            currents[idx] += td_bias
            
    return currents

def extract_lfp(traces: np.ndarray, cell_indices: List[int]):
    """Calculate mean trace for a set of cells."""
    return np.mean(traces[cell_indices, :], axis=0)

def detect_spikes(traces: np.ndarray, threshold: float = -20.0):
    """Detect spike times in traces."""
    spikes_dict = {}
    for i in range(traces.shape[0]):
        # Simple threshold crossing
        spike_mask = (traces[i, :-1] < threshold) & (traces[i, 1:] >= threshold)
        spikes_dict[i] = np.where(spike_mask)[0].tolist()
    return spikes_dict
