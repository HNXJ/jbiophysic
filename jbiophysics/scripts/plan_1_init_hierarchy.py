import jax
import jax.numpy as jnp
import jaxley as jx
import numpy as np
import os
import sys

# Add current directory to path
sys.path.insert(0, os.getcwd())

from jbiophysics.compose import NetBuilder, OptimizerFacade
from jbiophysics.systems.networks.omission_v1_column import build_v1_column
from jbiophysics.systems.networks.omission_two_column import build_omission_network

def build_11_area_hierarchy(seed=42):
    """
    Step 1: Hierarchical Assembly of the 11-area Macaque Hierarchy.
    Tiers:
      Tier 1 (Sensory): V1, V2
      Tier 2 (Mid-Order): V4, MT, MST, TEO, FST
      Tier 3 (Executive): FEF, PFC (mapped as 8B, 46, 9/46)
    """
    builder = NetBuilder(seed=seed)
    
    # Tier 1
    builder.add_population("E", n=20, cell_type="pyr", area="V1")
    builder.add_population("PV", n=5, cell_type="pv", area="V1")
    builder.add_population("SST", n=3, cell_type="sst", area="V1")
    
    builder.add_population("E", n=20, cell_type="pyr", area="V2")
    builder.add_population("PV", n=5, cell_type="pv", area="V2")
    
    # Tier 2 (Representative subset for prototype)
    mid_areas = ["V4", "MT", "MST", "TEO", "FST"]
    for area in mid_areas:
        builder.add_population("E", n=15, cell_type="pyr", area=area)
        builder.add_population("PV", n=4, cell_type="pv", area=area)
        
    # Tier 3
    builder.add_population("E", n=20, cell_type="pyr", area="FEF")
    builder.add_population("PV", n=5, cell_type="pv", area="FEF")
    
    builder.add_population("E", n=30, cell_type="pyr", area="PFC")
    builder.add_population("PV", n=8, cell_type="pv", area="PFC")
    builder.add_population("SST", n=4, cell_type="sst", area="PFC")

    # Intra-areal Connectivity (Canonical Motifs)
    for area in ["V1", "V2"] + mid_areas + ["FEF", "PFC"]:
        builder.connect("E", "PV", synapse="AMPA", p=0.2, area=area)
        builder.connect("PV", "E", synapse="GABAa", p=0.4, area=area)
        if area in ["V1", "PFC"]: # Areas with SST modeled
             builder.connect("SST", "E", synapse="GABAb", p=0.3, area=area)
             builder.connect("E", "SST", synapse="AMPA", p=0.1, area=area)

    # Inter-areal Feedforward (V1 -> V2 -> Tier 2 -> FEF -> PFC)
    builder.connect("V1.E", "V2.E", synapse="AMPA", p=0.1, g=0.5)
    builder.connect("V2.E", "V4.E", synapse="AMPA", p=0.1, g=0.5)
    builder.connect("V4.E", "FEF.E", synapse="AMPA", p=0.1, g=0.5)
    builder.connect("FEF.E", "PFC.E", synapse="AMPA", p=0.1, g=0.5)

    # Inter-areal Feedback (PFC -> FEF -> V4 -> V2 -> V1)
    builder.connect("PFC.E", "FEF.E", synapse="NMDA", p=0.05, g=0.3)
    builder.connect("FEF.E", "V4.E", synapse="NMDA", p=0.05, g=0.3)
    builder.connect("V4.E", "V2.E", synapse="NMDA", p=0.05, g=0.3)
    builder.connect("V2.E", "V1.E", synapse="NMDA", p=0.05, g=0.3)

    return builder

if __name__ == "__main__":
    builder = build_11_area_hierarchy()
    net = builder.build()
    
    # Initialize Optimizer for Step 8: Physiological Lag Tuning
    facade = (OptimizerFacade(net, method="AGSDR", lr=1e-3)
              .set_pop_offsets(builder.population_offsets)
              .set_constraints(firing_rate=(1.0, 10.0), kappa_max=0.1)
              .set_pop_constraints("V1.E", firing_rate=(2.0, 8.0)))
    
    print("Plan 1: Structural Hierarchy Initialized.")
    # result = facade.run(epochs=10) # Placeholder for actual run
