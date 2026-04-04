# Plan 4: Differentiable Sensitivity Analysis & Reverse Engineering

## Objective
To leverage JAX-native differentiability to map the causal influence of specific synapses on macroscopic cognitive signals (Prediction Error, MMN).

## Phase 1: Differentiable Gradient Mapping
**1. Differentiable Sensitivity Map:** 
Utilize `jax.grad` to compute $\frac{\partial \text{Omission\_Magnitude}}{\partial g_{syn}}$ for every synaptic type in the 11-area hierarchy.
- *Methodology:* Define a loss function that is simply the integrated V1 L2/3 activity during the omission window and differentiate with respect to all trainable parameters.
- *Result:* Produce a "Surprise Sensitivity Profile."

**2. MMN Profile Reverse-Engineering:** 
Use the `OptimizerFacade` to tune the **NMDA/AMPA ratio** in feedback pathways to match an empirical MMN temporal waveform.

## Phase 2: Hierarchical Propagation
**3. Laminar Offsets & PE Origins:** 
Compare spike-rate offsets between supragranular (L2/3) and infragranular (L5/6) populations within V1.
- *Methodology:* Use the `soft-argmax` lag estimator to track the propagation of the "Neural Surprise" from V1 L2/3 upward to V2 and TEO.

**4. Feedback Prediction Origin:** 
Trace the latency of the pre-omission Beta-band build-up to confirm it originates in PFC/FEF L5/6 and propagates downward.
