# src/jbiophysic/models/builders/hierarchy.py
import jaxley as jx
from jaxley.synapses import IonotropicSynapse
from .populations import construct_column
from jbiophysic.common.utils.logging import get_logger

logger = get_logger(__name__)

def build_cortical_hierarchy(n_areas: int = 11) -> jx.Network:
    """Axis 18: Authentic Jaxley inter-areal logic across multiple visual hierarchies."""
    logger.info(f"Building cortical hierarchy with {n_areas} areas")
    
    # 1. Build distinct area local networks
    areas = [construct_column() for _ in range(n_areas)]
    
    # 2. Combine into one macroscopic object
    brain = jx.Network(areas)
    
    # 3. Inter-Areal Connectivity (Sparse Routing)
    ff_synapse = IonotropicSynapse(gS=0.001)
    fb_synapse = IonotropicSynapse(gS=0.002)
    
    # Simple linear chain mapping V1 -> higher order
    for i in range(n_areas - 1):
        v1_pop = areas[i]
        v2_pop = areas[i+1]
        
        # Feedforward: L2/3 PC -> L4 / PC (Soma matching)
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
        
    logger.info("Cortical hierarchy assembly complete.")
    return brain

def build_11_area_hierarchy() -> jx.Network:
    """Legacy alias for build_cortical_hierarchy(n_areas=11)."""
    logger.info("Executing legacy alias: build_11_area_hierarchy")
    return build_cortical_hierarchy(n_areas=11)
