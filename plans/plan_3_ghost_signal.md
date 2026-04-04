# Plan 3: The "Ghost" Signal & Mechanics of Omission

## Objective
To isolate and reverse-engineer the "Ghost Signal"—the internally generated neural activity during a physical omission—and validate the VIP-SST disinhibition motifs.

## Phase 1: Tracing the Expectation
**1. "Ghost" Trace Quantization:** 
Extract virtual LFP and Pyramidal cell traces from V1 L2/3 during the omission window (1031ms - 1562ms).
- *Methodology:* Use the `ResultsReport` to slice the voltage traces and compute the mean firing rate density during the AXAB (Omit) condition.

**2. Prediction Error (PE) Isolation:** 
Subtract the AAAB (Predicted) activity from the AXAB (Omitted) activity in JAX to isolate the pure Negative Prediction Error (nPE).

## Phase 2: Testing Circuit Hypotheses
**3. SST-Dendritic Subtraction Failure:** 
Selectively suppress $g_{SST \to PYR}$ dendritic inhibition during the omission window.
- *Hypothesis:* missing SST-mediated cancellation should result in an unmasked, massive "Ghost Signal."

**4. VIP-Trigger Activation:** 
Test the VIP disinhibition hypothesis by injecting a precisely timed bias current into V1 VIP+ cells at $t=1031ms$.
- *Methodology:* Use `jx.integrate` with a time-varying `I_ext` vector delivered to the VIP population.
- *Goal:* Determine if VIP activation is the necessary and sufficient trigger for the omission response.

**5. Beta-Omission Link:** 
Verify if pre-omission Beta-band build-up (feedback) is a reliable predictor of the Ghost Signal magnitude.
