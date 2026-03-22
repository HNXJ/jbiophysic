import jax
import jax.numpy as jnp
import jaxley as jx
import numpy as np
from typing import Optional, Dict, List
from jbiophysics.core.mechanisms.models import Inoise, GradedAMPA, GradedGABAa, GradedGABAb
from jbiophysics.core.neurons.hh_cells import build_pyramidal_rs, build_pv_fs

def build_laminar_cells(num_superficial=40, num_mid=20, num_deep=40,
                        cell_types_ratio=None, spacing=20.0, seed=None):
    if seed is None: seed = int(np.random.randint(0, 2**31 - 1))
    np.random.seed(seed)
    if cell_types_ratio is None:
        cell_types_ratio = {'superficial': [0.8, 0.2], 'mid': [0.8, 0.2], 'deep': [0.8, 0.2]}
    layer_defs = [('superficial', num_superficial, 0.0, 400.0),
                  ('mid', num_mid, 400.0, 600.0), ('deep', num_deep, 600.0, 1000.0)]
    all_cells, cell_metadata = [], []
    for layer_name, count, z_start, z_end in layer_defs:
        ratios = cell_types_ratio[layer_name]; n_pyr = int(count * ratios[0])
        for i in range(count):
            cell = build_pyramidal_rs() if i < n_pyr else build_pv_fs()
            ctype = 'Pyr' if i < n_pyr else 'PV'
            theta = np.random.uniform(0, 2*np.pi)
            r = spacing * np.sqrt(np.random.uniform(0, 25))
            cell.xyz = np.array([r*np.cos(theta), r*np.sin(theta), np.random.uniform(z_start, z_end)])
            all_cells.append(cell)
            cell_metadata.append({'layer': layer_name, 'type': ctype})
    return all_cells, cell_metadata

def build_laminar_column(num_superficial=40, num_mid=20, num_deep=40,
                         cell_types_ratio=None, spacing=20.0, seed=None):
    cells, meta = build_laminar_cells(num_superficial, num_mid, num_deep, cell_types_ratio, spacing, seed)
    net = jx.Network(cells)
    from jaxley.connect import sparse_connect
    def get_indices(layer=None, ctype=None):
        return [i for i, m in enumerate(meta) if (layer is None or m['layer']==layer) and (ctype is None or m['type']==ctype)]
    for layer in ['superficial', 'mid', 'deep']:
        pyr = get_indices(layer, 'Pyr')
        if pyr: sparse_connect(net.cell(pyr).branch(0).loc(0.5), net.cell(pyr).branch(0).loc(0.5), GradedAMPA(g=0.5), p=0.1)
    mid_pyr, sup_pyr = get_indices('mid', 'Pyr'), get_indices('superficial', 'Pyr')
    if mid_pyr and sup_pyr:
        sparse_connect(net.cell(mid_pyr).branch(0).loc(0.5), net.cell(sup_pyr).branch(0).loc(0.5), GradedAMPA(g=1.0), p=0.2)
    for layer in ['superficial', 'mid', 'deep']:
        pv, pyr = get_indices(layer, 'PV'), get_indices(layer, 'Pyr')
        if pv and pyr:
            sparse_connect(net.cell(pv).branch(0).loc(0.0), net.cell(pyr).branch(0).loc(0.0), GradedGABAa(g=2.0), p=0.4)
    return net, meta
