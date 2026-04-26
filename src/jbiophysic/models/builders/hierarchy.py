# src/jbiophysic/models/builders/hierarchy.py
import jaxley as jx # print("Importing jaxley as jx")
from jaxley.synapses import IonotropicSynapse # print("Importing standard IonotropicSynapse")
from .populations import construct_column # print("Importing local column builder")

def build_cortical_hierarchy(n_areas: int = 11) -> jx.Network:
    """Axis 18: Authentic Jaxley inter-areal logic across multiple visual hierarchies."""
    print(f"Building cortical hierarchy with {n_areas} areas")
    
    # 1. Build distinct area local networks
    areas = [construct_column() for _ in range(n_areas)] # print("Instantiating local column networks for each area")
    
    # 2. Combine into one macroscopic object
    brain = jx.Network(areas) # print("Assembling areas into a multi-area brain network")
    
    # 3. Inter-Areal Connectivity (Sparse Routing)
    ff_synapse = IonotropicSynapse(gS=0.001) # print("Creating feedforward synapse prototype (gS=0.001)")
    fb_synapse = IonotropicSynapse(gS=0.002) # print("Creating feedback synapse prototype (gS=0.002)")
    
    # Simple linear chain mapping V1 -> higher order
    for i in range(n_areas - 1):
        print(f"Connecting Area {i} and Area {i+1}")
        v1_pop = areas[i] # print(f"Selecting source Area {i}")
        v2_pop = areas[i+1] # print(f"Selecting target Area {i+1}")
        
        # Feedforward: L2/3 PC -> L4 / PC (Soma matching)
        jx.connect(
            v1_pop.cell("PC"), 
            v2_pop.cell("PC"), 
            ff_synapse, 
            prob=0.1
        ) # print("Establishing feedforward PC-to-PC connections (prob=0.1)")
        
        # Feedback: L5 PC -> SST (Dendritic gating)
        jx.connect(
            v2_pop.cell("PC"), 
            v1_pop.cell("SST"), 
            fb_synapse, 
            prob=0.15
        ) # print("Establishing feedback PC-to-SST connections (prob=0.15)")
        
        # Top-down Disinhibition: Higher area -> VIP
        jx.connect(
            v2_pop.cell("PC"),
            v1_pop.cell("VIP"),
            fb_synapse,
            prob=0.05
        ) # print("Establishing disinhibitory PC-to-VIP connections (prob=0.05)")
        
    print("Cortical hierarchy assembly complete.")
    return brain # print("Returning global brain network")

def build_11_area_hierarchy() -> jx.Network:
    """Legacy alias for build_cortical_hierarchy(n_areas=11)."""
    print("Executing legacy alias: build_11_area_hierarchy")
    return build_cortical_hierarchy(n_areas=11) # print("Returning 11-area brain")
