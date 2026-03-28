import os
import sys
import jax
import jax.numpy as jnp
import jaxley as jx
import numpy as np
from typing import Any, Dict, List, Optional, Tuple, Union
import warnings

# --- Path Setup ---
sys.path.insert(0, '/Users/hamednejat/workspace/Repositories/AAE')
sys.path.insert(0, '/Users/hamednejat/workspace/Repositories/jbiophys')

from jbiophysics.core.mechanisms.models import GradedAMPA

def sort_cells_by_depth(cells: List[jx.Cell], meta: List[Dict]) -> Tuple[List[jx.Cell], List[Dict]]:
    """
    Sorts cells by their Z-axis coordinate (depth).
    Adds 1e-6 jitter to identical depths to ensure unique indexing.
    """
    # 1. Extract depths and indices
    depths = []
    for m in meta:
        xyz = m.get('xyz', np.zeros(3))
        depths.append(float(xyz[2]))
    
    depths = np.array(depths)
    
    # 2. Check for collisions and apply jitter
    unique_depths, counts = np.unique(depths, return_counts=True)
    if np.any(counts > 1):
        # Apply tiny deterministic jitter to ensure order
        jitter = np.linspace(0, 1e-6, len(depths))
        depths = depths + jitter
        
    # 3. Sort
    sort_idx = np.argsort(depths)
    sorted_cells = [cells[i] for i in sort_idx]
    sorted_meta = [meta[i] for i in sort_idx]
    
    # Update meta with sorted relative rank
    for i, m in enumerate(sorted_meta):
        m['relative_depth_rank'] = i
        
    return sorted_cells, sorted_meta

def merge_params_pytrees(net: jx.Network, params_list: List[List[Dict]]) -> List[Dict]:
    """
    Maps individual area PyTrees into the global parameter structure of the merged network.
    Assumes the global network has already been instantiated with the combined cells.
    """
    # JAXley's Network.get_parameters() returns a list of dictionaries.
    # We need to concatenate the arrays for each parameter key across groups.
    
    # 1. Identify all unique parameter keys across all areas
    keys = set()
    for area_params in params_list:
        for group in area_params:
            keys.update(group.keys())
            
    # 2. Reconstruct the global list of dictionaries
    # Typically: group 0 = edges, group 1 = nodes, etc.
    num_groups = len(params_list[0])
    global_params = []
    
    for g_idx in range(num_groups):
        group_dict = {}
        for key in keys:
            arrays = []
            for area_params in params_list:
                if g_idx < len(area_params) and key in area_params[g_idx]:
                    arrays.append(area_params[g_idx][key])
            
            if arrays:
                group_dict[key] = jnp.concatenate(arrays)
        global_params.append(group_dict)
        
    return global_params

def build_ensemble(
    area_configs: List[Dict],
    connectivity_matrix: np.ndarray,
    synapse_configs: Optional[Dict[Tuple[int, int], Any]] = None
) -> Tuple[jx.Network, List[Dict], List[Dict]]:
    """
    Builds a unified Network from multiple optimized areas.
    
    Inputs:
        area_configs: List of dicts with {'cells', 'meta', 'params'}
        connectivity_matrix: N x N matrix where entry (i, j) == 1 means Area i -> Area j.
        synapse_configs: Map of (src, tgt) index to Synapse instance.
    
    Returns:
        (MergedNetwork, MergedParams, MergedMetaData)
    """
    if synapse_configs is None:
        synapse_configs = {}

    all_cells = []
    all_meta = []
    all_params = []
    
    # 1. Process each area: Sort and Collect
    for i, config in enumerate(area_configs):
        cells = config['cells']
        meta = config['meta']
        params = config['params']
        
        # Ensure area label exists in meta
        for m in meta:
            m['area_index'] = i
            if 'area' not in m:
                m['area'] = f"Area_{i}"
        
        sorted_cells, sorted_meta = sort_cells_by_depth(cells, meta)
        
        all_cells.extend(sorted_cells)
        all_meta.extend(sorted_meta)
        all_params.append(params)
        
        # Store area-specific metadata for wiring
        config['sorted_indices'] = range(len(all_cells) - len(sorted_cells), len(all_cells))

    # 2. Instantiate Merged Network
    print(f"🧩 Merging {len(area_configs)} areas into Ensemble ({len(all_cells)} total neurons)...")
    net = jx.Network(all_cells)
    
    # 3. Apply Inter-Area Connectivity ("id to id")
    num_areas = len(area_configs)
    for src_idx in range(num_areas):
        for tgt_idx in range(num_areas):
            if connectivity_matrix[src_idx, tgt_idx] > 0:
                print(f"🔗 Wiring: Area {src_idx} -> Area {tgt_idx}...")
                
                # Get cell views for the populations
                src_inds = area_configs[src_idx]['sorted_indices']
                tgt_inds = area_configs[tgt_idx]['sorted_indices']
                
                # Synapse selection
                synapse = synapse_configs.get((src_idx, tgt_idx))
                if synapse is None:
                    warnings.warn(f"No synapse type specified for {src_idx}->{tgt_idx}. Defaulting to GradedAMPA(g=0.1).")
                    synapse = GradedAMPA(g=0.1)
                
                # "Id to Id" mapping: connect corresponding ranks
                # We use individual jx.connect calls for exact mapping
                for rank in range(min(len(src_inds), len(tgt_inds))):
                    jx.connect(
                        net.cell(src_inds[rank]).branch(0).loc(0.0), # Source soma
                        net.cell(tgt_inds[rank]).branch(1).loc(1.0), # Target dendrite (typical feedback/hierarchical)
                        synapse
                    )

    # 4. Merge Parameters
    merged_params = merge_params_pytrees(net, all_params)
    
    print("✅ Ensemble Construction Complete.")
    return net, merged_params, all_meta
