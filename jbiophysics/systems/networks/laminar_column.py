import jax
import jax.numpy as jnp
import jaxley as jx
import numpy as np
from typing import Optional, Dict, List
from jbiophysics.core.mechanisms.models import Inoise, GradedAMPA, GradedGABAa, GradedGABAb
from jbiophysics.core.neurons.hh_cells import build_pyramidal_rs, build_pv_fs

def build_laminar_cells(
    num_superficial: int = 40,
    num_mid: int = 20,
    num_deep: int = 40,
    cell_types_ratio: Optional[Dict] = None,
    spacing: float = 20.0,
    seed: Optional[int] = None
):
    """
    Builds raw cell objects and metadata for a laminar cortical column.
    """
    if seed is None:
        seed = int(np.random.randint(0, 2**31 - 1))
    np.random.seed(seed)
    
    if cell_types_ratio is None:
        cell_types_ratio = {
            'superficial': [0.8, 0.2],
            'mid': [0.8, 0.2],
            'deep': [0.8, 0.2]
        }

    layer_definitions = [
        ('superficial', num_superficial, 0.0, 400.0),
        ('mid', num_mid, 400.0, 600.0),
        ('deep', num_deep, 600.0, 1000.0)
    ]

    all_cells = []
    cell_metadata = []

    for layer_name, count, z_start, z_end in layer_definitions:
        ratios = cell_types_ratio[layer_name]
        n_pyr = int(count * ratios[0])
        
        for i in range(count):
            if i < n_pyr:
                cell = build_pyramidal_rs()
                ctype = 'Pyr'
            else:
                cell = build_pv_fs()
                ctype = 'PV'
            
            theta = np.random.uniform(0, 2*np.pi)
            r = spacing * np.sqrt(np.random.uniform(0, 5**2)) 
            x = r * np.cos(theta)
            y = r * np.sin(theta)
            z = np.random.uniform(z_start, z_end)
            
            cell.xyz = np.array([x, y, z])
            all_cells.append(cell)
            cell_metadata.append({'layer': layer_name, 'type': ctype, 'z': z})

    return all_cells, cell_metadata

def build_laminar_column(
    num_superficial: int = 40,
    num_mid: int = 20,
    num_deep: int = 40,
    cell_types_ratio: Optional[Dict] = None,
    spacing: float = 20.0,
    seed: Optional[int] = None
):
    """
    Builds a laminar column and initializes it as a jx.Network.
    """
    cells, meta = build_laminar_cells(num_superficial, num_mid, num_deep, cell_types_ratio, spacing, seed)
    net = jx.Network(cells)
    
    # Intra-column connections
    from jaxley.connect import sparse_connect
    
    def get_indices(layer=None, ctype=None):
        return [idx for idx, m in enumerate(meta) if (layer is None or m['layer'] == layer) and (ctype is None or m['type'] == ctype)]

    for layer in ['superficial', 'mid', 'deep']:
        pyr_inds = get_indices(layer, 'Pyr')
        if pyr_inds:
            sparse_connect(net.cell(pyr_inds).branch(0).loc(0.5), net.cell(pyr_inds).branch(0).loc(0.5), GradedAMPA(g=0.5), p=0.1)

    mid_pyr = get_indices('mid', 'Pyr')
    sup_pyr = get_indices('superficial', 'Pyr')
    if mid_pyr and sup_pyr:
        sparse_connect(net.cell(mid_pyr).branch(0).loc(0.5), net.cell(sup_pyr).branch(0).loc(0.5), GradedAMPA(g=1.0), p=0.2)

    for layer in ['superficial', 'mid', 'deep']:
        pv_inds = get_indices(layer, 'PV')
        pyr_inds = get_indices(layer, 'Pyr')
        if pv_inds and pyr_inds:
            sparse_connect(net.cell(pv_inds).branch(0).loc(0.0), net.cell(pyr_inds).branch(0).loc(0.0), GradedGABAa(g=2.0), p=0.4)

    return net, meta

if __name__ == "__main__":
    print("Testing Laminar Column Builder...")
    net, meta = build_laminar_column(num_superficial=10, num_mid=5, num_deep=10)
    print(f"Network built with {len(list(net.cells))} cells across 3 layers.")
