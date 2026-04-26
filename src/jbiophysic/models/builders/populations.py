# src/jbiophysic/models/builders/populations.py
from jbiophysic.common.utils.logging import get_logger

logger = get_logger(__name__)

import jaxley as jx
from jbiophysic.core.mechanisms.channels.hh_base import HH

def build_pyramidal_cell():
    """Axis 18: Authentic Jaxley morphological instantiation for Layer 5/23 PC."""
    logger.info("Building Pyramidal Cell morphology")
    # In Jaxley 0.5.0+, Cell takes a list of Branch objects.
    # Each Branch contains one or more Compartments.
    soma = jx.Branch(ncomp=1)
    apical = jx.Branch(ncomp=1)
    basal = jx.Branch(ncomp=1)
    
    cell = jx.Cell([soma, apical, basal], parents=[-1, 0, 0])
    
    # Insert specific ionic conductances
    cell.insert(HH())
    
    # Scale leak conductances per branch
    cell.branch(0).set("gl", 0.0003)
    cell.branch(1).set("gl", 0.0001)
    cell.branch(2).set("gl", 0.0001)
    
    return cell

def build_interneuron(cell_type="PV"):
    """Axis 18: Specific interneuron morphologies."""
    logger.info(f"Building interneuron: {cell_type}")
    # For a single-compartment cell, we can pass a single Branch object (not a list).
    # If a single object is passed, parents must be None.
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
    logger.info("Constructing full cortical column with population labels")
    n_pc = 200
    n_pv, n_sst, n_vip = 40, 40, 20
    
    # Build populations
    pc_cells = [build_pyramidal_cell() for _ in range(n_pc)]
    pv_cells = [build_interneuron("PV") for _ in range(n_pv)]
    sst_cells = [build_interneuron("SST") for _ in range(n_sst)]
    vip_cells = [build_interneuron("VIP") for _ in range(n_vip)]
    
    # Combine into one network
    all_cells = pc_cells + pv_cells + sst_cells + vip_cells
    column_net = jx.Network(all_cells)
    
    # Explicitly label groups for later selection (e.g., .cell("PC"))
    column_net.add_type("PC", list(range(0, n_pc)))
    column_net.add_type("PV", list(range(n_pc, n_pc + n_pv)))
    column_net.add_type("SST", list(range(n_pc + n_pv, n_pc + n_pv + n_sst)))
    column_net.add_type("VIP", list(range(n_pc + n_pv + n_sst, n_pc + n_pv + n_sst + n_vip)))
    
    return column_net
