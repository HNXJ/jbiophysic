# Plan 4: Differentiable Sensitivity Analysis & Reverse Engineering

## Objective
To fully leverage the JAX-native differentiability of `jbiophysics` to map the causal influence of specific synapses on macroscopic cognitive signals (Prediction Error, MMN), transforming the model from descriptive to explanatory.

## Phase 1: Differentiable Gradient Mapping
**1. Differentiable Sensitivity Map (Task 4):** 
Utilize `jax.grad` to compute the exact gradient of the **Omission Signal Magnitude** (defined as the mean firing rate or integrated LFP power in V1 L2/3 at $t=1031ms$) with respect to every single inter-areal and intra-areal synaptic conductance in the 11-area hierarchy.
- *Execution:* Evaluate $\frac{\partial \text{Omission\_Magnitude}}{\partial g_{Feedback}}$ vs. $\frac{\partial \text{Omission\_Magnitude}}{\partial g_{Local\_Inhibition}}$.
- *Target Measurement:* Produce a "Sensitivity Profile" identifying the absolute critical "control knobs" of surprise. Are top-down NMDA conductances more impactful than local VIP disinhibition?

**2. MMN Profile Reverse-Engineering (Task 7):** 
The Mismatch Negativity (MMN) is a canonical ERP for omission/oddball detection. Define a target temporal profile matching an empirical MMN waveform (derived from MUAe envelopes).
- *Execution:* Use `jax.grad` and the `OptimizerFacade` to exclusively tune the **NMDA/AMPA ratio** in the specific feedback motif (e.g., PFC -> V1).
- *Target Measurement:* Prove that the specific temporal delay and decay of the MMN can be fully explained by the slow kinetics of NMDA-mediated feedback.

## Phase 2: Hierarchical Propagation Mapping
**3. Laminar Offsets & PE Origins (Task 10):** 
Extract and compare the high-resolution spike-rate offsets between supragranular (L2/3 - PE origin) and infragranular (L5/6 - Prediction origin) populations within the V1 column during the Omission window.
- *Target Measurement:* Verify the "Hierarchical Prediction Error" hypothesis by tracking the latency of the "Neural Surprise" as it travels upward. Does the PE signal reliably appear in V1 L2/3 *before* it propagates to V2 L4 and subsequently to Tier 2 areas?

**4. The Feedback "Prediction" Origin:** 
Conversely, trace the latency of the pre-omission Beta-band build-up. It should originate in Tier 3 (PFC/FEF) L5/6 and propagate downward to V1, arriving precisely before $t=1031ms$.
