# Cortex network builder

`jbiophysic.networks.cortex.make_cortex_network` creates a layered 3-D cortical volume
specification for three execution families:

- `izhikevich`: point-neuron metadata and source-based AMPA/GABA synapse table.
- `tfne`: geometry and TFNE source metadata for field/CSD/LFP forward modeling.
- `tfne-izhikevich`: a hybrid source-coupled bridge with explicit Izhikevich-current to ampere calibration.

## Required inputs

```python
from jbiophysic.networks.cortex import make_cortex_network

net = make_cortex_network(
    XYZ_mm=[0.5, 0.5, 1.5],
    N=1000,
    Ls=[0.4, 0.2, 0.4],
    Ld=[
        [70, 10, 10, 10],
        [75, 20, 4, 1],
        [75, 15, 9, 1],
    ],
    model_family="tfne-izhikevich",
    plasticity_coefficient=0.25,
    seed=42,
)
```

`XYZ_mm` is `[X, Y, Z]` in millimetres. Layers span the z-axis. `Ls` is normalized to
layer fractions. `Ld` rows are `[E, PV, SST, VIP]` percentages or fractions. Neuron counts
are allocated with largest-remainder rounding so the total is exactly `N`.

## Non-overlap rule

Positions are sampled randomly but must satisfy the requested centre-to-centre separation
`min_distance_um`. If the density is impossible, the builder fails explicitly instead of
silently overlapping neurons.

## Default receptor rule

Source E neurons project AMPA. Source PV/SST/VIP neurons project GABA. Cell types also carry
waveform labels and Izhikevich parameter presets.

## TFNE safety rule

The hybrid model does not relabel native Izhikevich current units as SI amperes. It records
`izh_current_to_ampere_scale` in the TFNE metadata and keeps TFNE positions/radii in SI units.
