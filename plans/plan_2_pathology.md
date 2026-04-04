# Plan 2: Modeling the Pathological State & Interneuron Deficits

## Objective
To systematically introduce and optimize pathological constraints representing the Interneuron Deficit Hypothesis and map the resulting changes to network stability and rhythmogenesis using AGSDR v2.

## Phase 1: Inducing Core Deficits
**1. PV-Deficit Implementation:** 
Systematically scale down $g_{PV \to E}$ (perisomatic inhibition) by 30-50% in L2/3 across the hierarchy. 
- *Methodology:* Use `builder.make_trainable` to isolate these conductances and apply a soft-constraint penalty in the loss function to force them toward the deficit target.
- *Measurement:* Quantify the collapse of stimulus-evoked Gamma power (41 Hz) and the increase in baseline firing rate variance.

**2. CB/SST Compensation:** 
Following the PV deficit, use **AGSDR v2** to selectively increase $g_{SST \to E}$ to stabilize global firing rates. 
- *Methodology:* Optimize the $E \to SST$ and $SST \to E$ loop to minimize the firing rate error while maintaining the 30% PV deficit.
- *Measurement:* Analyze the shift from perisomatic to dendritic inhibition dominance.

**3. NMDA Hypofunction:** 
Reduce `GradedNMDA` maximum conductance in top-down pathways (PFC $\to$ V1). 
- *Methodology:* Map the $\Delta$ Beta power (13-30 Hz) as a function of $g_{NMDA}$ reduction.

## Phase 2: Landscape Mapping
**4. Kappa Landscape Mapping:** 
Run a multi-dimensional sweep of PV and SST deficit severities. 
- *Target:* Map the **"Hallucinogenic Valley"**—the region where the system transitions from asynchronous ($Kappa < 0.1$) to pathological hypersynchrony ($Kappa > 0.2$).

**5. Spectral Motif Matching (SSS):** 
Force the optimizer to find parameter sets that reproduce the "Ketamine-model" power spectrum (high Beta/Gamma ratio shift) using the log-PSD matching loss.
