# src/jbiophysic/models/builders/populations.py
import jaxley as jx
from jbiophysic.core.mechanisms.channels.hh_base import HH

def build_pyramidal_cell():
    """Axis 18: Authentic Jaxley morphological instantiation for Layer 5/23 PC."""
    print("Building Pyramidal Cell morphology")
    comp = jx.Compartment()
    branch1 = jx.Compartment()
    branch2 = jx.Compartment()
    
    cell = jx.Cell([comp, branch1, branch2], parents=[-1, 0, 0])
    
    # Insert specific ionic conductances
    cell.insert(HH())
    
    # Scale conductances
    cell.branch(0).set("gl_HH", 0.0003)
    cell.branch(1).set("gl_HH", 0.0001)
    cell.branch(2).set("gl_HH", 0.0001)
    
    return cell

def build_interneuron(cell_type="PV"):
    """Axis 18: Specific interneuron morphologies."""
    print(f"Building interneuron: {cell_type}")
    cell = jx.Cell([jx.Compartment()])
    cell.insert(HH())
    
    if cell_type == "PV":
        cell.set("gk_HH", 0.036 * 1.5)
    elif cell_type == "SST":
        cell.set("gl_HH", 0.0001)
    elif cell_type == "VIP":
        cell.set("gl_HH", 0.0002)
        
    return cell

def construct_column():
    """Assembles a local cortical column from explicit Jaxley Cell objects."""
    print("Constructing full cortical column")
    n_pc = 200
    n_pv, n_sst, n_vip = 40, 40, 20
    
    pc_population = jx.Network([build_pyramidal_cell() for _ in range(n_pc)])
    pv_population = jx.Network([build_interneuron("PV") for _ in range(n_pv)])
    sst_population = jx.Network([build_interneuron("SST") for _ in range(n_sst)])
    vip_population = jx.Network([build_interneuron("VIP") for _ in range(n_vip)])
    
    # Group them natively in Jaxley
    column_net = jx.Network([pc_population, pv_population, sst_population, vip_population])
    return column_net
