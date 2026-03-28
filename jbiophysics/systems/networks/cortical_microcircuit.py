import jax
import jax.numpy as jnp
import jaxley as jx
import numpy as np
from typing import Optional
from core.mechanisms.models import Inoise, GradedAMPA, GradedGABAa, GradedGABAb

def build_cortical_column(name_prefix: str, num_e: int=36, num_pv: int=8, num_sst: int=6, num_vip: int=4, seed: Optional[int] = None):
    if seed is None:
        seed = int(np.random.randint(0, 2**31 - 1))
    np.random.seed(seed)
    
    comp_soma = jx.Compartment()
    comp_dend = jx.Compartment()
    
    e_cells = [jx.Cell([comp_soma, comp_dend], parents=[-1, 0]) for _ in range(num_e)]
    for cell in e_cells:
        cell.radius, cell.length = 1.0, 100.0
        cell.insert(jx.channels.HH())
        cell.insert(Inoise(initial_amp_noise=0.02, initial_tau=20.0))

    branch_in = jx.Branch(comp_soma, ncomp=1)
    pv_cells = [jx.Cell(branch_in, parents=[-1]) for _ in range(num_pv)]
    sst_cells = [jx.Cell(branch_in, parents=[-1]) for _ in range(num_sst)]
    vip_cells = [jx.Cell(branch_in, parents=[-1]) for _ in range(num_vip)]

    for cell in pv_cells + sst_cells + vip_cells:
        cell.radius, cell.length = 1.0, 10.0
        cell.insert(jx.channels.HH())
        cell.insert(Inoise(initial_amp_noise=0.02, initial_tau=10.0))

    net = jx.Network(e_cells + pv_cells + sst_cells + vip_cells)
    from jaxley.connect import sparse_connect

    sparse_connect(net.cell(range(num_e)).branch(0).loc(0.0), net.cell('all').branch(0).loc(0.0), GradedAMPA(g=2.0), p=0.5)
    sparse_connect(net.cell(range(num_e, num_e+num_pv)).branch(0).loc(0.0), net.cell(range(num_e)).branch(0).loc(0.0), GradedGABAa(g=5.0), p=0.6)
    sparse_connect(net.cell(range(num_e+num_pv, num_e+num_pv+num_sst)).branch(0).loc(0.0), net.cell(range(num_e)).branch(1).loc(1.0), GradedGABAb(g=1.0), p=0.4)
    sparse_connect(net.cell(range(num_e+num_pv+num_sst, num_e+num_pv+num_sst+num_vip)).branch(0).loc(0.0), net.cell(range(num_e+num_pv, num_e+num_pv+num_sst)).branch(0).loc(0.0), GradedGABAa(g=2.0), p=0.5)

    return net
