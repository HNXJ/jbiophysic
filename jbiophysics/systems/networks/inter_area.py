import jax
import jax.numpy as jnp
import jaxley as jx
import numpy as np
from typing import List, Tuple, Dict
from jbiophysics.core.mechanisms.models import GradedAMPA, GradedGABAa

def connect_cortical_areas(source_config: Dict, target_config: Dict, p_ff=0.2, p_fb=0.1, g_ff=0.5, g_fb=0.3):
    """
    Builds two cortical columns and connects them using canonical laminar motifs.
    """
    from jbiophysics.systems.networks.laminar_column import build_laminar_cells
    from jaxley.connect import sparse_connect
    
    # 1. Build Cells for both areas
    src_cells, src_meta = build_laminar_cells(**source_config)
    tgt_cells, tgt_meta = build_laminar_cells(**target_config)
    
    # 2. Combine and initialize Network
    all_cells = src_cells + tgt_cells
    net = jx.Network(all_cells)
    
    num_source = len(src_cells)
    
    # Helper to get global indices
    def get_source_inds(layer, ctype):
        return [i for i, m in enumerate(src_meta) if m['layer'] == layer and m['type'] == ctype]
    
    def get_target_inds(layer, ctype):
        return [num_source + i for i, m in enumerate(tgt_meta) if m['layer'] == layer and m['type'] == ctype]

    # --- Intra-Area Connections ---
    def connect_intra(meta, offset):
        def get_inds(layer=None, ctype=None):
            return [offset + idx for idx, m in enumerate(meta) if (layer is None or m['layer'] == layer) and (ctype is None or m['type'] == ctype)]
        
        for layer in ['superficial', 'mid', 'deep']:
            pyr_inds = get_inds(layer, 'Pyr')
            if pyr_inds:
                sparse_connect(net.cell(pyr_inds).branch(0).loc(0.5), net.cell(pyr_inds).branch(0).loc(0.5), GradedAMPA(g=0.5), p=0.1)
        
        mid_pyr = get_inds('mid', 'Pyr')
        sup_pyr = get_inds('superficial', 'Pyr')
        if mid_pyr and sup_pyr:
            sparse_connect(net.cell(mid_pyr).branch(0).loc(0.5), net.cell(sup_pyr).branch(0).loc(0.5), GradedAMPA(g=1.0), p=0.2)

        for layer in ['superficial', 'mid', 'deep']:
            pv_inds = get_inds(layer, 'PV')
            pyr_inds = get_inds(layer, 'Pyr')
            if pv_inds and pyr_inds:
                sparse_connect(net.cell(pv_inds).branch(0).loc(0.0), net.cell(pyr_inds).branch(0).loc(0.0), GradedGABAa(g=2.0), p=0.4)

    connect_intra(src_meta, 0)
    connect_intra(tgt_meta, num_source)

    # --- Inter-Area Canonical Motifs ---
    # Feedforward (Source L2/3 -> Target L4)
    src_sup = get_source_inds('superficial', 'Pyr')
    tgt_mid = get_target_inds('mid', 'Pyr')
    if src_sup and tgt_mid:
        sparse_connect(net.cell(src_sup).branch(0).loc(0.5), 
                       net.cell(tgt_mid).branch(0).loc(0.5), 
                       GradedAMPA(g=g_ff), p=p_ff)

    # Feedback (Target L5/6 -> Source L1/L2)
    tgt_deep = get_target_inds('deep', 'Pyr')
    src_sup_fb = get_source_inds('superficial', 'Pyr')
    if tgt_deep and src_sup_fb:
        sparse_connect(net.cell(tgt_deep).branch(0).loc(0.5), 
                       net.cell(src_sup_fb).branch(1).loc(1.0), 
                       GradedAMPA(g=g_fb), p=p_fb)

    return net, (src_meta, tgt_meta)

if __name__ == "__main__":
    from jbiophysics.systems.networks.laminar_column import build_laminar_column
    print("Building Two-Area Network...")
    net_low, meta_low = build_laminar_column(num_superficial=10, num_mid=5, num_deep=10, seed=1)
    net_high, meta_high = build_laminar_column(num_superficial=10, num_mid=5, num_deep=10, seed=2)
    
    # We combine them manually using the connection logic
    combined_net = connect_cortical_areas(list(net_low.cells), meta_low, list(net_high.cells), meta_high)
    print(f"Combined Network built with {len(list(combined_net.cells))} cells.")
