import jax
import jax.numpy as jnp
import jaxley as jx
import numpy as np
from typing import Optional, Dict, Any, List, Tuple
import sys

# --- Path Setup ---
# Assumes Repositories root
sys.path.insert(0, '/Users/hamednejat/workspace/Repositories/AAE')
sys.path.insert(0, '/Users/hamednejat/workspace/Repositories/jbiophys')

from core.mechanisms.models import Inoise, GradedAMPA, GradedGABAa, GradedGABAb, make_synapses_independent

def get_cylinder_positions(n, z_mean, z_std, radius=100.0):
    """Generates 3D coordinates in a cylinder."""
    r = radius * jnp.sqrt(np.random.rand(n))
    theta = 2 * jnp.pi * np.random.rand(n)
    z = np.random.normal(z_mean, z_std, n)
    return jnp.stack([r * jnp.cos(theta), r * jnp.sin(theta), z], axis=-1)

def build_biophysical_cells(
    num_e: int = 36, 
    num_pv: int = 8, 
    num_sst: int = 6, 
    num_vip: int = 4, 
    seed: Optional[int] = None
) -> Tuple[List[jx.Cell], Dict[str, Any]]:
    """Builds raw cell objects for a single column with realistic 3D positions."""
    if seed is None:
        seed = int(np.random.randint(0, 2**31 - 1))
    np.random.seed(seed)
    
    comp_soma = jx.Compartment()
    comp_dend = jx.Compartment()
    
    # E-cells (2 compartments)
    e_cells = [jx.Cell([comp_soma, comp_dend], parents=[-1, 0]) for _ in range(num_e)]
    e_pos = get_cylinder_positions(num_e, z_mean=-500, z_std=150)
    for i, cell in enumerate(e_cells):
        cell.radius = 10.0; cell.length = 500.0; cell.xyz = e_pos[i]
        cell.insert(jx.channels.HH())
        cell.insert(Inoise(initial_amp_noise=np.random.uniform(0.1, 0.2), initial_tau=20.0))

    # Interneurons (1 compartment)
    pv_cells = [jx.Cell(jx.Branch(comp_soma, ncomp=1), parents=[-1]) for _ in range(num_pv)]
    sst_cells = [jx.Cell(jx.Branch(comp_soma, ncomp=1), parents=[-1]) for _ in range(num_sst)]
    vip_cells = [jx.Cell(jx.Branch(comp_soma, ncomp=1), parents=[-1]) for _ in range(num_vip)]

    pv_pos = get_cylinder_positions(num_pv, z_mean=-400, z_std=50)
    sst_pos = get_cylinder_positions(num_sst, z_mean=-600, z_std=50)
    vip_pos = get_cylinder_positions(num_vip, z_mean=-200, z_std=50)

    for i, cell in enumerate(pv_cells):
        cell.radius = 5.0; cell.length = 10.0; cell.xyz = pv_pos[i]
        cell.insert(jx.channels.HH()); cell.insert(Inoise(initial_amp_noise=0.1, initial_tau=10.0))
    for i, cell in enumerate(sst_cells):
        cell.radius = 4.0; cell.length = 10.0; cell.xyz = sst_pos[i]
        cell.insert(jx.channels.HH()); cell.insert(Inoise(initial_amp_noise=0.1, initial_tau=10.0))
    for i, cell in enumerate(vip_cells):
        cell.radius = 3.0; cell.length = 10.0; cell.xyz = vip_pos[i]
        cell.insert(jx.channels.HH()); cell.insert(Inoise(initial_amp_noise=0.1, initial_tau=10.0))

    all_cells = e_cells + pv_cells + sst_cells + vip_cells
    counts = {"E": num_e, "PV": num_pv, "SST": num_sst, "VIP": num_vip, "total": len(all_cells)}
    meta = [{'type': 'Pyr', 'layer': 'L2/3', 'xyz': c.xyz} for c in e_cells] + \
           [{'type': 'PV', 'layer': 'L4', 'xyz': c.xyz} for c in pv_cells] + \
           [{'type': 'SST', 'layer': 'L5', 'xyz': c.xyz} for c in sst_cells] + \
           [{'type': 'VIP', 'layer': 'L1', 'xyz': c.xyz} for c in vip_cells]

    return all_cells, {"counts": counts, "meta": meta}

def build_biophysical_column(
    name: str, 
    num_e: int = 36, 
    num_pv: int = 8, 
    num_sst: int = 6, 
    num_vip: int = 4, 
    seed: Optional[int] = None
) -> Tuple[jx.Network, Dict[str, Any]]:
    """Builds a single column with realistic 3D positions and biophysical sizes."""
    cells, info = build_biophysical_cells(num_e, num_pv, num_sst, num_vip, seed)
    net = jx.Network(cells)
    apply_column_internal_connectivity(net, 0, num_e, num_pv, num_sst, num_vip)
    return net, info

def apply_column_internal_connectivity(net, offset, num_e, num_pv, num_sst, num_vip):
    """Applies internal connections to a specific column within a merged network."""
    from jaxley.connect import sparse_connect
    total = num_e + num_pv + num_sst + num_vip
    e_indices = [offset + i for i in range(num_e)]
    pv_indices = [offset + num_e + i for i in range(num_pv)]
    sst_indices = [offset + num_e + num_pv + i for i in range(num_sst)]
    vip_indices = [offset + num_e + num_pv + num_sst + i for i in range(num_vip)]

    # Internal E->E (minus self)
    for i in e_indices:
        others = [j for j in e_indices if j != i]
        if others:
            sparse_connect(net.cell(i).branch(0).loc(0.0), net.cell(others).branch(0).loc(0.0), GradedAMPA(g=0.5, tauD_AMPA=2.0), p=1.0)

    # E -> Interneurons
    if pv_indices: sparse_connect(net.cell(e_indices).branch(0).loc(0.0), net.cell(pv_indices).branch(0).loc(0.0), GradedAMPA(g=0.5), p=1.0)
    if sst_indices: sparse_connect(net.cell(e_indices).branch(0).loc(0.0), net.cell(sst_indices).branch(0).loc(0.0), GradedAMPA(g=0.5), p=1.0)
    
    # Internal Inhibitory
    if pv_indices: sparse_connect(net.cell(pv_indices).branch(0).loc(0.0), net.cell(e_indices).branch(0).loc(0.0), GradedGABAa(g=8.0), p=1.0)
    if sst_indices: sparse_connect(net.cell(sst_indices).branch(0).loc(0.0), net.cell(e_indices).branch(1).loc(1.0), GradedGABAb(g=1.0), p=1.0)
    if vip_indices and sst_indices: sparse_connect(net.cell(vip_indices).branch(0).loc(0.0), net.cell(sst_indices).branch(0).loc(0.0), GradedGABAa(g=2.0), p=1.0)

def build_three_area_mscz(
    num_e: int = 8, 
    num_pv: int = 2, 
    num_sst: int = 0, 
    num_vip: int = 0, 
    seed: Optional[int] = 42
) -> Tuple[jx.Network, Dict[str, Any]]:
    """Builds three hierarchical columns (Low -> Mid -> High)."""
    # Build raw cells
    cells_l, info_l = build_biophysical_cells(num_e=num_e, num_pv=num_pv, num_sst=num_sst, num_vip=num_vip, seed=seed)
    cells_m, info_m = build_biophysical_cells(num_e=num_e, num_pv=num_pv, num_sst=num_sst, num_vip=num_vip, seed=seed+1)
    cells_h, info_h = build_biophysical_cells(num_e=num_e, num_pv=num_pv, num_sst=num_sst, num_vip=num_vip, seed=seed+2)
    
    # Initialize single network
    all_cells = cells_l + cells_m + cells_h
    net = jx.Network(all_cells)
    
    # Offsets and indices
    l_total = info_l["counts"]["total"]
    m_total = info_m["counts"]["total"]
    h_total = info_h["counts"]["total"]
    
    l_offset = 0
    m_offset = l_total
    h_offset = l_total + m_total
    
    # Indices for FF/FB
    l_e = range(l_offset, l_offset + num_e)
    m_e = range(m_offset, m_offset + num_e)
    m_pv = range(m_offset + num_e, m_offset + num_e + num_pv)
    h_e = range(h_offset, h_offset + num_e)
    h_pv = range(h_offset + num_e, h_offset + num_e + num_pv)

    # 1. Apply Internal Connectivity
    apply_column_internal_connectivity(net, l_offset, num_e, num_pv, num_sst, num_vip)
    apply_column_internal_connectivity(net, m_offset, num_e, num_pv, num_sst, num_vip)
    apply_column_internal_connectivity(net, h_offset, num_e, num_pv, num_sst, num_vip)

    # 2. Hierarchical Connections
    from jaxley.connect import sparse_connect
    # FF: Low -> Mid
    sparse_connect(net.cell(l_e).branch(0).loc(0.0), net.cell(m_e).branch(0).loc(0.0), GradedAMPA(g=0.2), p=1.0)
    sparse_connect(net.cell(l_e).branch(0).loc(0.0), net.cell(m_pv).branch(0).loc(0.0), GradedAMPA(g=0.2), p=1.0)
    
    # FF: Mid -> High
    sparse_connect(net.cell(m_e).branch(0).loc(0.0), net.cell(h_e).branch(0).loc(0.0), GradedAMPA(g=0.2), p=1.0)
    sparse_connect(net.cell(m_e).branch(0).loc(0.0), net.cell(h_pv).branch(0).loc(0.0), GradedAMPA(g=0.2), p=1.0)
    
    # FB: High -> Mid
    sparse_connect(net.cell(h_e).branch(0).loc(0.0), net.cell(m_e).branch(1).loc(1.0), GradedAMPA(g=0.1), p=1.0)
    
    # FB: Mid -> Low
    sparse_connect(net.cell(m_e).branch(0).loc(0.0), net.cell(l_e).branch(1).loc(1.0), GradedAMPA(g=0.1), p=1.0)

    # Enforce independent synapses
    make_synapses_independent(net, "gAMPA")
    make_synapses_independent(net, "gGABAa")
    if num_sst > 0: make_synapses_independent(net, "gGABAb")

    info = {
        "l_info": info_l, "m_info": info_m, "h_info": info_h, 
        "total_cells": len(all_cells),
        "offsets": {"Low": l_offset, "Mid": m_offset, "High": h_offset},
        "meta": info_l["meta"] + info_m["meta"] + info_h["meta"]
    }
    return net, info

def build_hierarchical_mscz(
    num_e: int = 36, 
    seed: Optional[int] = 42
) -> Tuple[jx.Network, Dict[str, Any]]:
    """Builds two hierarchical columns (Lower -> Higher)."""
    # Build raw cells
    cells_l, info_l = build_biophysical_cells(num_e=num_e, num_pv=8, num_sst=6, num_vip=4, seed=seed)
    cells_h, info_h = build_biophysical_cells(num_e=num_e, num_pv=8, num_sst=6, num_vip=4, seed=seed+1)
    
    # Initialize single network
    all_cells = cells_l + cells_h
    net = jx.Network(all_cells)
    
    l_total = info_l["counts"]["total"]
    l_e_indices = range(info_l["counts"]["E"])
    h_offset = l_total
    h_e_indices = range(h_offset, h_offset + info_h["counts"]["E"])
    h_pv_indices = range(h_offset + info_h["counts"]["E"], h_offset + info_h["counts"]["E"] + info_h["counts"]["PV"])

    # Re-apply internal connections
    apply_column_internal_connectivity(net, 0, info_l["counts"]["E"], 8, 6, 4)
    apply_column_internal_connectivity(net, h_offset, info_h["counts"]["E"], 8, 6, 4)
    
    # Hierarchical Connections
    from jaxley.connect import sparse_connect
    # FF: Lower E -> Higher E & PV
    sparse_connect(net.cell(l_e_indices).branch(0).loc(0.0), net.cell(h_e_indices).branch(0).loc(0.0), GradedAMPA(g=0.2), p=1.0)
    sparse_connect(net.cell(l_e_indices).branch(0).loc(0.0), net.cell(h_pv_indices).branch(0).loc(0.0), GradedAMPA(g=0.2), p=1.0)
    # FB: Higher E -> Lower E Dend
    sparse_connect(net.cell(h_e_indices).branch(0).loc(0.0), net.cell(l_e_indices).branch(1).loc(1.0), GradedAMPA(g=0.1), p=1.0)

    # Enforce independent synapses
    make_synapses_independent(net, "gAMPA")
    make_synapses_independent(net, "gGABAa")
    make_synapses_independent(net, "gGABAb")

    info = {"l_info": info_l, "h_info": info_h, "total_cells": len(all_cells), "meta": info_l["meta"] + info_h["meta"]}
    return net, info

def run_pre_tuning_sweep(net: jx.Network, target_range: Tuple[float, float] = (1.0, 100.0)):
    """Performs coarse sweeps to find physiological baseline."""
    print("🌊 Running Constructor Pre-tuning Sweep...")
    dt = 0.1
    net.cell('all').branch(0).loc(0.0).record('v')
    
    # Final check
    voltages = jx.integrate(net, t_max=1000.0, delta_t=dt)
    threshold = -20.0
    spikes = (voltages[:, :-1] < threshold) & (voltages[:, 1:] >= threshold)
    fr = jnp.sum(spikes) / (len(list(net.cells)) * 1.0)
    print(f"   Constructor Baseline FR: {fr:.2f} Hz")
    return fr
