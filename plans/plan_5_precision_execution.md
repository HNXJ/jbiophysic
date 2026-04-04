# Plan 5: Precision Quenching & Execution Strategy

## Objective
To evaluate the "Precision Scaling" hypothesis by analyzing neural variability post-omission, and to execute the complete "Gemini CLI" workflow generating a definitive 3D error surface of the cortex.

## Phase 1: Precision Quenching Audit
**1. The Post-Omission Window ($P3$):** 
Focus analysis on the $P3$ stimulus window ($t=2062ms$) that immediately follows an omission ($X2$) versus a standard stimulus ($P2$).
- *Hypothesis:* Predictive coding suggests that highly surprising events (omissions) cause the brain to upregulate the "precision" (gain) of subsequent sensory inputs to rapidly update its internal model.

**2. Quenching Analysis (Task 20):** 
Calculate the Fano factor or trial-to-trial variance of the Pyramidal cell firing rates during the $P3$ window.
- *Target Measurement:* If the "Precision Scaling" hypothesis holds, the neural variability (noise) should be significantly "quenched" (reduced) in the $P3$ following an omission compared to the $P3$ following a predicted stimulus, indicating heightened sensory gain driven by PV+ interneurons.

## Phase 2: The Error Surface Mapping
**3. 3D Landscape Generation:** 
Combine the results of the baseline optimization (Plan 1), pathological sweeps (Plan 2), and sensitivity mapping (Plan 4).
- *Execution:* Generate a comprehensive 3D surface plot mapping:
  - Axis X: Severity of Interneuron Deficit (e.g., % reduction in PV/SST).
  - Axis Y: Strength of Top-Down Feedback ($g_{NMDA}$).
  - Axis Z (Height): Magnitude of the "Ghost Signal" (Prediction Error).

**4. Identifying the "Hallucinogenic Valley":** 
Locate the specific coordinates on this 3D surface where the prediction error collapses entirely (failure of reality testing) or where internal predictions dominate sensory reality (hallucination regime).

## Phase 3: Final Integration & Execution Workflow
**5. Automated Sequence Execution:** 
Implement the 21-Step Masterclass via the `jbiophysics` API:
1. Initialize the `NetBuilder` hierarchy.
2. Run `OptimizerFacade` for baseline tuning (AAAB).
3. Perturb parameters and run the AXAB omission sequence.
4. Execute `jax.grad` for sensitivity analysis.
5. Export the resulting matrices to the `ResultsReport` for integration into the Plotly/Reveal.js dashboard.

This final plan synthesizes all biophysical mechanisms, pathological perturbations, and differentiable analyses into a single, cohesive executable pipeline.
