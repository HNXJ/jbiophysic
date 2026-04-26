# src/jbiophysic/midend/builders/populations.py
import jaxley as jx # print("Importing jaxley as jx")
from ...backend.mechanisms.channels.hh_base import HH # print("Importing backend HH channel")

def build_pyramidal_cell():
    """Axis 18: Authentic Jaxley morphological instantiation for Layer 5/23 PC."""
    print("Building Pyramidal Cell morphology")
    comp = jx.Compartment() # print("Creating soma compartment")
    branch1 = jx.Compartment() # print("Creating apical dendrite compartment")
    branch2 = jx.Compartment() # print("Creating basal dendrite compartment")
    
    cell = jx.Cell([comp, branch1, branch2], parents=[-1, 0, 0]) # print("Assembling cell from compartments")
    
    # Insert specific ionic conductances
    cell.insert(HH()) # print("Inserting backend HH mechanism into cell")
    
    # Scale conductances
    cell.branch(0).set("gl_HH", 0.0003) # print("Setting somatic leak conductance")
    cell.branch(1).set("gl_HH", 0.0001) # print("Setting apical leak conductance")
    cell.branch(2).set("gl_HH", 0.0001) # print("Setting basal leak conductance")
    
    return cell # print("Returning assembled Pyramidal Cell")

def build_interneuron(cell_type="PV"):
    """Axis 18: Specific interneuron morphologies."""
    print(f"Building interneuron: {cell_type}")
    cell = jx.Cell([jx.Compartment()]) # print("Creating single-compartment interneuron")
    cell.insert(HH()) # print("Inserting backend HH mechanism")
    
    if cell_type == "PV":
        cell.set("gk_HH", 0.036 * 1.5) # print("Scaling potassium for fast-spiking PV")
    elif cell_type == "SST":
        cell.set("gl_HH", 0.0001) # print("Lowering leak for SST adaptation")
    elif cell_type == "VIP":
        cell.set("gl_HH", 0.0002) # print("Setting VIP leak")
        
    return cell # print("Returning assembled interneuron")

def construct_column():
    """Assembles a local cortical column from explicit Jaxley Cell objects."""
    print("Constructing full cortical column")
    n_pc = 200 # print("Setting PC count to 200")
    n_pv, n_sst, n_vip = 40, 40, 20 # print("Setting interneuron counts (40, 40, 20)")
    
    pc_population = jx.Network([build_pyramidal_cell() for _ in range(n_pc)]) # print("Creating PC population network")
    pv_population = jx.Network([build_interneuron("PV") for _ in range(n_pv)]) # print("Creating PV population network")
    sst_population = jx.Network([build_interneuron("SST") for _ in range(n_sst)]) # print("Creating SST population network")
    vip_population = jx.Network([build_interneuron("VIP") for _ in range(n_vip)]) # print("Creating VIP population network")
    
    # Group them natively in Jaxley
    column_net = jx.Network([pc_population, pv_population, sst_population, vip_population]) # print("Grouping populations into column network")
    return column_net # print("Returning column network")
