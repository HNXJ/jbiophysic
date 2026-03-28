import jax
import jax.numpy as jnp
import jaxley as jx
import numpy as np
from typing import Optional, Dict, Any, List, Tuple
import sys

# --- Path Setup ---
sys.path.insert(0, '/Users/hamednejat/workspace/Repositories/AAE')
sys.path.insert(0, '/Users/hamednejat/workspace/Repositories/jbiophys')

from jbiophysics.core.mechanisms.models import Inoise, GradedAMPA, GradedGABAa, GradedGABAb, GradedNMDA, make_synapses_independent

def get_cylinder_positions(n, z_mean, z_std, radius=100.0):
    """Generates 3D coordinates in a cylinder."""
    r = radius * jnp.sqrt(np.random.rand(n))
    theta = 2 * jnp.pi * np.random.rand(n)
    z = np.random.normal(z_mean, z_std, n)
    return jnp.stack([r * jnp.cos(theta), r * jnp.sin(theta), z], axis=-1)

def build_biophysical_cells(
    num_total: int = 200,
    ei_ratio: float = 0.75,
    nmda_subset_ratio: float = 0.1,
    seed: Optional[int] = None
) -> Tuple[List[jx.Cell], Dict[str, Any]]:
    """
    Builds raw cell objects for a 6-layered cortical column.
    75% Excitatory, 25% Inhibitory.
    10% of E-cells are marked for NMDA output.
    """
    if seed is None:
        seed = int(np.random.randint(0, 2**31 - 1))
    np.random.seed(seed)
    
    num_e = int(num_total * ei_ratio)
    num_i = num_total - num_e
    num_nmda_e = int(num_e * nmda_subset_ratio)
    
    comp_soma = jx.Compartment()
    comp_dend = jx.Compartment()
    
    # 1. Distribute E-cells (Bimodal: L2/3 and L5/6)
    e_cells = []
    e_meta = []
    # L2/3 (Superficial)
    n_e_sup = int(num_e * 0.6)
    e_pos_sup = get_cylinder_positions(n_e_sup, z_mean=-300, z_std=100)
    # L5/6 (Deep)
    n_e_deep = num_e - n_e_sup
    e_pos_deep = get_cylinder_positions(n_e_deep, z_mean=-800, z_std=100)
    
    e_pos = jnp.concatenate([e_pos_sup, e_pos_deep], axis=0)
    
    # Randomly select NMDA subset
    nmda_indices = np.random.choice(range(num_e), size=num_nmda_e, replace=False)
    
    for i in range(num_e):
        cell = jx.Cell([comp_soma, comp_dend], parents=[-1, 0])
        cell.radius = 10.0; cell.length = 500.0; cell.xyz = e_pos[i]
        cell.insert(jx.channels.HH())
        # Lower noise for baseline control
        cell.insert(Inoise(initial_amp_noise=np.random.uniform(0.05, 0.1), initial_tau=20.0))
        e_cells.append(cell)
        e_meta.append({
            'type': 'Pyr', 
            'layer': 'Superficial' if i < n_e_sup else 'Deep', 
            'xyz': e_pos[i],
            'has_nmda': i in nmda_indices
        })

    # 2. Distribute I-cells (PV in L4, SST in L2/5, VIP in L1/2)
    i_cells = []
    i_meta = []
    n_pv = int(num_i * 0.4)
    n_sst = int(num_i * 0.3)
    n_vip = num_i - n_pv - n_sst
    
    pv_pos = get_cylinder_positions(n_pv, z_mean=-500, z_std=50) # L4
    sst_pos = get_cylinder_positions(n_sst, z_mean=-600, z_std=150) # L2/L5
    vip_pos = get_cylinder_positions(n_vip, z_mean=-100, z_std=50) # L1/2
    
    for i, pos in enumerate(pv_pos):
        cell = jx.Cell(jx.Branch(comp_soma, ncomp=1), parents=[-1])
        cell.radius = 5.0; cell.length = 10.0; cell.xyz = pos
        cell.insert(jx.channels.HH()); cell.insert(Inoise(initial_amp_noise=0.05, initial_tau=10.0))
        i_cells.append(cell); i_meta.append({'type': 'PV', 'layer': 'L4', 'xyz': pos})
        
    for i, pos in enumerate(sst_pos):
        cell = jx.Cell(jx.Branch(comp_soma, ncomp=1), parents=[-1])
        cell.radius = 4.0; cell.length = 10.0; cell.xyz = pos
        cell.insert(jx.channels.HH()); cell.insert(Inoise(initial_amp_noise=0.05, initial_tau=10.0))
        i_cells.append(cell); i_meta.append({'type': 'SST', 'layer': 'L2/5', 'xyz': pos})
        
    for i, pos in enumerate(vip_pos):
        cell = jx.Cell(jx.Branch(comp_soma, ncomp=1), parents=[-1])
        cell.radius = 3.0; cell.length = 10.0; cell.xyz = pos
        cell.insert(jx.channels.HH()); cell.insert(Inoise(initial_amp_noise=0.05, initial_tau=10.0))
        i_cells.append(cell); i_meta.append({'type': 'VIP', 'layer': 'L1/2', 'xyz': pos})

    all_cells = e_cells + i_cells
    all_meta = e_meta + i_meta
    counts = {"E": num_e, "I": num_i, "NMDA_E": num_nmda_e, "total": len(all_cells)}
    
    return all_cells, {"counts": counts, "meta": all_meta}

def apply_cortical_internal_connectivity(net: jx.Network, offset: int, meta: List[Dict]):
    """Applies canonical microcircuitry with AMPA, NMDA, and GABAa."""
    from jaxley.connect import sparse_connect
    
    e_inds = [offset + i for i, m in enumerate(meta) if m['type'] == 'Pyr']
    nmda_e_inds = [offset + i for i, m in enumerate(meta) if m['type'] == 'Pyr' and m.get('has_nmda')]
    pv_inds = [offset + i for i, m in enumerate(meta) if m['type'] == 'PV']
    sst_inds = [offset + i for i, m in enumerate(meta) if m['type'] == 'SST']
    vip_inds = [offset + i for i, m in enumerate(meta) if m['type'] == 'VIP']
    all_inds = range(offset, offset + len(meta))

    # 1. E -> All (AMPA) - Lower conductance
    sparse_connect(net.cell(e_inds).branch(0).loc(0.0), net.cell(all_inds).branch(0).loc(0.0), GradedAMPA(g=0.2), p=0.1)
    
    # 2. NMDA Subset E -> E (NMDA)
    if nmda_e_inds:
        sparse_connect(net.cell(nmda_e_inds).branch(0).loc(0.0), net.cell(e_inds).branch(0).loc(0.0), GradedNMDA(g=0.1), p=0.2)

    # 3. PV -> E Soma (GABAa)
    if pv_inds:
        sparse_connect(net.cell(pv_inds).branch(0).loc(0.0), net.cell(e_inds).branch(0).loc(0.0), GradedGABAa(g=5.0), p=0.4)

    # 4. SST -> E Dendrite (GABAb)
    if sst_inds:
        sparse_connect(net.cell(sst_inds).branch(0).loc(0.0), net.cell(e_inds).branch(1).loc(1.0), GradedGABAb(g=1.0), p=0.3)

    # 5. VIP -> SST (GABAa disinhibition)
    if vip_inds and sst_inds:
        sparse_connect(net.cell(vip_inds).branch(0).loc(0.0), net.cell(sst_inds).branch(0).loc(0.0), GradedGABAa(g=2.0), p=0.5)

def build_v1_v2_v4_hierarchy(
    area_params_list: Optional[List[List[Dict]]] = None,
    seed: int = 42
) -> Tuple[jx.Network, Dict[str, Any]]:
    """
    Builds the 3-area V1->V2->V4 hierarchy.
    If area_params_list is provided, it maps them to the merged network.
    """
    # 1. Build cells
    cells_v1, info_v1 = build_biophysical_cells(seed=seed)
    cells_v2, info_v2 = build_biophysical_cells(seed=seed+1)
    cells_v4, info_v4 = build_biophysical_cells(seed=seed+2)
    
    net = jx.Network(cells_v1 + cells_v2 + cells_v4)
    
    # Offsets
    v1_off, v2_off, v4_off = 0, 200, 400
    
    # 2. Internal connectivity
    apply_cortical_internal_connectivity(net, v1_off, info_v1['meta'])
    apply_cortical_internal_connectivity(net, v2_off, info_v2['meta'])
    apply_cortical_internal_connectivity(net, v4_off, info_v4['meta'])
    
    # 3. Hierarchical wiring (Markov 2014)
    from jaxley.connect import sparse_connect
    def get_e_sup(off, meta): return [off + i for i, m in enumerate(meta) if m['type'] == 'Pyr' and m['layer'] == 'Superficial']
    def get_e_deep(off, meta): return [off + i for i, m in enumerate(meta) if m['type'] == 'Pyr' and m['layer'] == 'Deep']
    def get_l4_soma(off, meta): return [off + i for i, m in enumerate(meta) if m['layer'] == 'L4']
    def get_l1_dend(off, meta): return [off + i for i, m in enumerate(meta) if m['layer'] == 'L1/2']

    # FF: V1 -> V2, V2 -> V4, V1 -> V4
    for src, tgt, s_info, t_info in [(v1_off, v2_off, info_v1, info_v2), (v2_off, v4_off, info_v2, info_v4), (v1_off, v4_off, info_v1, info_v4)]:
        src_e = get_e_sup(src, s_info['meta'])
        tgt_l4 = get_l4_soma(tgt, t_info['meta'])
        sparse_connect(net.cell(src_e).branch(0).loc(0.0), net.cell(tgt_l4).branch(0).loc(0.0), GradedAMPA(g=0.2), p=0.1)

    # FB: V4 -> V2, V2 -> V1, V4 -> V1
    for src, tgt, s_info, t_info in [(v4_off, v2_off, info_v4, info_v2), (v2_off, v1_off, info_v2, info_v1), (v4_off, v1_off, info_v4, info_v1)]:
        src_e = get_e_deep(src, s_info['meta'])
        tgt_l1 = get_l1_dend(tgt, t_info['meta'])
        sparse_connect(net.cell(src_e).branch(0).loc(0.0), net.cell(tgt_l1).branch(0).loc(0.0), GradedAMPA(g=0.1), p=0.1)

    # 4. Enforce independent synapses
    make_synapses_independent(net, "gAMPA")
    make_synapses_independent(net, "gGABAa")
    make_synapses_independent(net, "gNMDA")
    # GABAb from SST
    make_synapses_independent(net, "gGABAb")

    # 5. Load Parameters if provided
    params = None
    if area_params_list:
        from .ensemble import merge_params_pytrees
        params = merge_params_pytrees(net, area_params_list)

    info = {
        "v1": info_v1, "v2": info_v2, "v4": info_v4,
        "meta": info_v1['meta'] + info_v2['meta'] + info_v4['meta']
    }
    return net, params, info

def run_pre_tuning_sweep(net: jx.Network, target_range: Tuple[float, float] = (1.0, 100.0)):
    """Performs coarse sweeps to find physiological baseline."""
    print("🌊 Running Constructor Pre-tuning Sweep...")
    dt = 0.1
    net.cell('all').branch(0).loc(0.0).record('v')
    voltages = jx.integrate(net, t_max=1000.0, delta_t=dt)
    threshold = -20.0
    spikes = (voltages[:, :-1] < threshold) & (voltages[:, 1:] >= threshold)
    fr = jnp.sum(spikes) / (len(list(net.cells)) * 1.0)
    print(f"   Constructor Baseline FR: {fr:.2f} Hz")
    return fr
