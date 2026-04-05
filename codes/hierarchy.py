# jbiophysics/codes/hierarchy.py
import jaxley as jx
from jaxley.synapses import IonotropicSynapse
from codes.neurons import construct_column

def build_cortical_hierarchy(n_areas=11):
    """Axis 18: Authentic Jaxley inter-areal logic across 11 visual hierarchies."""
    # 1. Build distinct area local networks
    areas = [construct_column() for _ in range(n_areas)]
    
    # 2. Combine into one macroscopic object
    brain = jx.Network(areas)
    
    # 3. Inter-Areal Connectivity Matrix (Sparse Routing)
    ff_synapse = IonotropicSynapse(gS=0.001)
    fb_synapse = IonotropicSynapse(gS=0.002)
    
    # Simple linear chain mapping V1 -> higher order
    for i in range(n_areas - 1):
        v1_pop = areas[i]
        v2_pop = areas[i+1]
        
        # Feedforward: L2/3 PC -> L4 / PC (Soma matching)
        # Using probabilistic connectivity 
        jx.connect(
            v1_pop.cell("PC"), 
            v2_pop.cell("PC"), 
            ff_synapse, 
            prob=0.1
        )
        
        # Feedback: L5 PC -> SST (Dendritic gating)
        jx.connect(
            v2_pop.cell("PC"), 
            v1_pop.cell("SST"), 
            fb_synapse, 
            prob=0.15
        )
        
        # Top-down Disinhibition: Higher area -> VIP
        jx.connect(
            v2_pop.cell("PC"),
            v1_pop.cell("VIP"),
            fb_synapse,
            prob=0.05
        )
        
    # Must compile the Jaxley structure before returning
    # jaxley compiles via step or integrate, so we just return the raw Network
    return brain
