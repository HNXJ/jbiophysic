# Cortex network builder

`jbiophysic.networks.cortex.make_cortex_network` creates a high-fidelity, layered 3-D cortical volume
specification. This tool is designed to bridge the gap between abstract connectivity and 
biophysical reality, supporting three primary execution families:

- `izhikevich`: Optimized for point-neuron simulations. It generates positions and a 
  distance-dependent, source-based AMPA/GABA synapse table.
- `tfne`: Tailored for forward-field modeling. It provides the geometry and TFNE source 
  metadata necessary for calculating extracellular fields, Current Source Density (CSD), 
  and Local Field Potentials (LFP).
- `tfne-izhikevich`: A sophisticated hybrid bridge. It couples the temporal precision 
  of Izhikevich neurons with the spatial depth of TFNE, including explicit calibration 
  factors to map phenomenological Izhikevich currents to physical SI amperes.

## Detailed Configuration

The builder uses a "geometry-first" approach. You specify the physical dimensions and 
the desired neuron density, and the engine handles the spatial allocation and connectivity.

```python
from jbiophysic.networks.cortex import make_cortex_network

# Example: Building a 1.5mm deep cortical column
net = make_cortex_network(
    XYZ_mm=[0.5, 0.5, 1.5],  # 0.5x0.5mm surface area, 1.5mm depth
    N=1000,                  # Target population size
    Ls=[0.4, 0.2, 0.4],      # Layer thickness fractions (e.g., L2/3, L4, L5/6)
    Ld=[                     # Cell-type densities [E, PV, SST, VIP] per layer
        [70, 10, 10, 10],    # Superficial
        [75, 20, 4, 1],      # Middle (Input layer)
        [75, 15, 9, 1],      # Deep
    ],
    model_family="tfne-izhikevich",
    plasticity_coefficient=0.25,
    seed=123,                # Reproducible stochastic generation
)
```

### Parameter Breakdown

- `XYZ_mm`: Physical dimensions in millimetres. The Z-axis is always treated as the depth 
  axis for laminar boundaries.
- `Ls`: A list of fractions (summing to 1) defining the relative thickness of each layer.
- `Ld`: A matrix where each row corresponds to a layer defined in `Ls`. Each row defines 
  the relative density of Excitatory (E), Parvalbumin (PV), Somatostatin (SST), and 
  Vasoactive Intestinal Peptide (VIP) neurons.
- `plasticity_coefficient`: Scales the initial weight of synapses that are marked as plastic.

## Spatial Constraints: The Non-Overlap Rule

To ensure biological plausibility, the builder employs a rejection-sampling algorithm. 
Each neuron position is sampled randomly but must maintain a minimum centre-to-centre 
distance (`min_distance_um`). If the requested population `N` is too high for the 
given volume and `min_distance_um`, the builder will throw an explicit error.

## Connectivity: Distance-Dependent Receptors

The connectivity is derived from Euclidean distance. 
- **Excitatory (E) sources** project to targets via **AMPA** receptors.
- **Inhibitory (PV, SST, VIP) sources** project via **GABA** receptors.

The probability of connection follows an exponential decay with distance, governed by the 
`connection_length_constant_mm` parameter.

## TFNE Integration and Calibration

For the `tfne-izhikevich` family, the builder automatically computes:
1. `source_positions_m`: Neuron positions converted to SI metres.
2. `source_radii_m`: Standardized volumetric source radii for CSD calculation.
3. `izh_current_to_ampere_scale`: A scaling factor (default `1e-12`) used to convert 
   dimensionless Izhikevich currents into physical currents for the TFNE kernel.
