# Plan 2: Modeling the Pathological State & Interneuron Deficits

## Objective
To systematically introduce and optimize pathological constraints representing the Interneuron Deficit Hypothesis (Schizophrenia/Ketamine models) and map the resulting changes to network stability and rhythmogenesis.

## Phase 1: Inducing Core Deficits
**1. PV-Deficit Implementation (H1):** 
To model the primary defect hypothesized in schizophrenia, systematically scale down the $g_{PV \to E}$ (perisomatic inhibition) by 30% in L2/3 across all areas. 
- *Target Measurement:* Observe and quantify the collapse of stimulus-evoked Gamma power (Task #1) and the widening of the temporal tuning curve (reduced precision).

**2. CB/SST Compensation (H2):** 
Following the primary PV deficit, the network must compensate to prevent catastrophic seizure. Use **AGSDR v2** to selectively increase $g_{SST \to E}$ or local recurrent inhibition to stabilize the firing rates. 
- *Target Measurement:* This should model the compensatory shift toward dendritic inhibition, potentially leading to enhanced Beta-band activity and a rigid, "over-predicted" cortical state.

**3. NMDA Hypofunction (H8):** 
Reduce the `GradedNMDA` maximum conductance in the deep-layer feedback pathways (e.g., **FEF/PFC $\to$ V1**). 
- *Target Measurement:* Because top-down predictions rely on slow NMDA kinetics, this should specifically blunt the "Top-Down Prediction" signal, impairing the network's ability to generate strong Beta oscillations during expectation windows.

## Phase 2: Landscape Mapping & Stability Analysis
**4. Kappa Landscape Mapping:** 
Run a comprehensive, multi-dimensional sweep of PV and SST deficit severities using the `OptimizerFacade`. 
- *Target Measurement:* Map the **"Critical Points"** in the parameter space where the system undergoes a phase transition from healthy, asynchronous operation ($Kappa \approx 0$) to pathological, hallucinatory hypersynchrony ($Kappa > 0.2$) (Task #5).

**5. Spectral Motif Matching (SSS):** 
Utilize the **Spectral Similarity Score** within the optimization loop. Provide the optimizer with an empirical LFP power spectrum derived from a "Ketamine-model" macaque. 
- *Target Measurement:* Force AGSDR to find the exact combination of E/I synaptic shifts required to reproduce the pathological power spectrum (Task #9).

**6. Pathological Stable Regimes:** 
Investigate if the system, when forced to maintain a 5Hz baseline firing rate under severe PV/NMDA constraints, discovers novel, pathological "stable states" (e.g., heavily reliant on Inoise variance or over-weighted recurrent excitation) that mirror the "Unstable Attractor" hypothesis of schizophrenia.
