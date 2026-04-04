# Plan 1: Structural Alignment & Healthy Baseline Optimization

## Objective
To establish a biologically accurate, stable, and differentiable 11-area cortical hierarchy (V1 to PFC) that perfectly aligns with the temporal dynamics and baseline physiological metrics (firing rates, lag, and synchrony) of the empirical monkey data.

## Phase 1: Structural & Biophysical Foundation
**1. Hierarchical Assembly:** 
Use the `NetBuilder` to instantiate the 11-area macaque hierarchy (V1, V2, V4, MT, MST, TEO, FST, FEF, PFC) across three functional tiers: Sensory (Tier 1), Mid-Order (Tier 2), and Executive (Tier 3). Each area must follow the hierarchical `Area.Population` indexing pattern to maintain spatial tracking of activity.

**2. Multi-Compartment Specification:** 
Define the Pyramidal ($E$) cells in the sensory and mid-order tiers with distinct dendritic and perisomatic compartments. This is critical for enabling the **SST-mediated dendritic subtraction** mechanism required later for prediction error computation. Utilize the `SafeHH` primitive (`name="HH"`) to ensure numerical stability during high-intensity integration.

**3. Interneuron Population Mapping:** 
Assign PV+, SST+, and VIP+ populations to each area. Their connectivity must strictly align with the canonical motifs described by Bastos et al. (2015): 
- PV+ targets Pyramidal somas (fast gain control).
- SST+ targets Pyramidal apical dendrites (top-down prediction cancellation).
- VIP+ targets SST+ cells (contextual disinhibition).

**4. Synaptic Kinetic Initialization:** 
Set precise time constants for synaptic mechanisms to reflect their role in canonical rhythms:
- $GABA_A$ (fast, tau ~5ms) and AMPA (fast, tau ~5ms) to drive feedforward "Error" frequencies (Gamma).
- $NMDA$ (slow, voltage-gated, tau ~100ms) to drive inter-areal feedback "Prediction" frequencies (Alpha/Beta).

## Phase 2: Paradigm & Stimulus Calibration
**5. Code 101.0 Alignment:** 
Calibrate the global simulation clock to the monkey's empirical **P1 onset** ($t=0$). The 500ms to 1000ms pre-stimulus buffer must be used exclusively for baseline stabilization, allowing the `Inoise` processes to settle into a natural 1-5Hz spontaneous firing rate.

**6. Luminance-Matched Background:** 
Ensure the "gray screen" inter-stimulus intervals (ISI) and the "Omission" windows have identical background luminance levels. This is modeled by ensuring strictly zero bottom-up (`BU`) sensory drive during these windows, removing all external triggers.

**7. Sequential Block Implementation:** 
Encode the 12 condition groups (e.g., **AAAB**, **AXAB**, **RRRR**) as perfectly aligned, differentiable 1D input vectors (`T,` shape) delivered exclusively to the $L4$ populations of V1.

## Phase 3: The "Healthy" Baseline Optimization
**8. Physiological Lag Tuning:** 
Deploy the `OptimizerFacade` using **AGSDR v2**. Adjust the $g_{AMPA}$ synaptic weights of the feedforward pathway until the V1 population firing rate peak aligns perfectly with the empirical **40–60ms** post-stimulus lag observed in the monkey data.

**9. PING/ING Rhythm Calibration:** 
Optimize the local $g_{PV 	o E}$ and $g_{E 	o PV}$ conductances to achieve a stable, physiological **40 Hz Gamma rhythm** during attended stimulus periods ($P1$ through $P4$). This is validated using the differentiable log-PSD matching (`SSS`) component of the loss function.

**10. Baseline Kappa Stabilization:** 
Calculate **Fleiss' Kappa** across the 11 areas during the fixation period. The objective function must strongly penalize Kappa values $> 0.1$ to ensure the "Healthy" model maintains asynchronous stability ($Kappa \approx 0$) before any stimulus is presented.
