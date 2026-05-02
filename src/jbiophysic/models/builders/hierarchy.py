# src/jbiophysic/models/builders/hierarchy.py
import jaxley as jx
from jaxley.synapses import IonotropicSynapse
from .populations import construct_column
from jbiophysic.common.utils.logging import get_logger

logger = get_logger(__name__)

def build_cortical_hierarchy(n_areas: int = 11) -> jx.Network:
    """Inter-areal connectivity across multiple visual areas."""
    logger.info(f"Building cortical hierarchy with {n_areas} areas")
    
    # 1. Collect all cells from all areas
    # construct_column() now returns a list of cells.
    all_cells = []
    for _ in range(n_areas):
        all_cells.extend(construct_column())
    
    # 2. Combine into one macroscopic object
    brain = jx.Network(all_cells)
    
    # Re-apply population and area groups on the flat network
    # Each area has 300 cells: 200 PC, 40 PV, 40 SST, 20 VIP
    cells_per_area = 300
    for i in range(n_areas):
        base = i * cells_per_area
        brain.cell(list(range(base, base + 200))).add_to_group("PC")
        brain.cell(list(range(base + 200, base + 240))).add_to_group("PV")
        brain.cell(list(range(base + 240, base + 280))).add_to_group("SST")
        brain.cell(list(range(base + 280, base + 300))).add_to_group("VIP")
        # Add area-specific groups for inter-areal routing
        area_name = f"Area_{i}"
        brain.cell(list(range(base, base + cells_per_area))).add_to_group(area_name)
    
    # 3. Inter-Areal Connectivity (Point-to-Point mapping to avoid internal broadcasting errors)
    ff_synapse = IonotropicSynapse()
    fb_synapse = IonotropicSynapse()
    
    # Simple linear chain mapping V1 -> higher order
    cells_per_area = 300
    for i in range(n_areas - 1):
        base_i = i * cells_per_area
        base_next = (i + 1) * cells_per_area
        
        # Feedforward: Area i PC -> Area i+1 PC (Connect first 200 PCs one-to-one)
        for j in range(200):
            jx.connect(
                brain.cell(base_i + j).branch(0).comp(0), 
                brain.cell(base_next + j).branch(0).comp(0), 
                ff_synapse
            )
        
        # Feedback: Area i+1 PC -> Area i SST (Connect first 40 PCs to 40 SSTs)
        for j in range(40):
            jx.connect(
                brain.cell(base_next + j).branch(0).comp(0), 
                brain.cell(base_i + 240 + j).branch(0).comp(0), 
                fb_synapse
            )
        
        # Top-down Disinhibition: Higher area -> VIP (Connect first 20 PCs to 20 VIPs)
        for j in range(20):
            jx.connect(
                brain.cell(base_next + 40 + j).branch(0).comp(0), 
                brain.cell(base_i + 280 + j).branch(0).comp(0), 
                fb_synapse
            )
        
    logger.info("Cortical hierarchy assembly complete.")
    return brain

def build_11_area_hierarchy() -> jx.Network:
    """Legacy alias for build_cortical_hierarchy(n_areas=11)."""
    logger.info("Executing legacy alias: build_11_area_hierarchy")
    return build_cortical_hierarchy(n_areas=11)
