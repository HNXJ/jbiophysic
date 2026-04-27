# src/jbiophysic/models/builders/populations.py
import jaxley as jx
from jbiophysic.core.mechanisms.channels.hh_base import HH
from jbiophysic.common.utils.logging import get_logger

logger = get_logger(__name__)

def build_pyramidal_cell():
    """Morphological instantiation for Layer 5/23 PC using Jaxley."""
    soma = jx.Branch(ncomp=1)
    apical = jx.Branch(ncomp=1)
    basal = jx.Branch(ncomp=1)
    cell = jx.Cell([soma, apical, basal], parents=[-1, 0, 0])
    cell.insert(HH())
    cell.branch(0).set("gl", 0.0003)
    cell.branch(1).set("gl", 0.0001)
    cell.branch(2).set("gl", 0.0001)
    return cell

def build_interneuron(cell_type="PV"):
    """Interneuron morphologies (PV/SST/VIP)."""
    cell = jx.Cell(jx.Branch(ncomp=1))
    cell.insert(HH())
    if cell_type == "PV":
        cell.set("gk", 0.036 * 1.5)
    elif cell_type == "SST":
        cell.set("gl", 0.0001)
    elif cell_type == "VIP":
        cell.set("gl", 0.0002)
    return cell

def construct_column():
    """Assembles a local cortical column with explicit population labels."""
    logger.info("Constructing cortical column populations (PC, PV, SST, VIP)")
    n_pc, n_pv, n_sst, n_vip = 200, 40, 40, 20
    
    # Instantiate cells
    pc_cells = [build_pyramidal_cell() for _ in range(n_pc)]
    pv_cells = [build_interneuron("PV") for _ in range(n_pv)]
    sst_cells = [build_interneuron("SST") for _ in range(n_sst)]
    vip_cells = [build_interneuron("VIP") for _ in range(n_vip)]
    
    # Combine into a single macroscopic network
    all_cells = pc_cells + pv_cells + sst_cells + vip_cells
    column_net = jx.Network(all_cells)
    
    # Axis 18: Mandatory population labeling for hierarchy selectors
    # This enables usage like network.cell("PC") in inter-areal logic.
    column_net.add_type("PC", list(range(0, n_pc)))
    column_net.add_type("PV", list(range(n_pc, n_pc + n_pv)))
    column_net.add_type("SST", list(range(n_pc + n_pv, n_pc + n_pv + n_sst)))
    column_net.add_type("VIP", list(range(n_pc + n_pv + n_sst, n_pc + n_pv + n_sst + n_vip)))
    
    logger.info(f"Column built with {len(all_cells)} cells across 4 populations.")
    return column_net
