# Plan 3: The "Ghost" Signal & Mechanics of Omission

## Objective
To isolate and reverse-engineer the "Ghost Signal"—the internally generated neural activity during a physical omission (gray screen)—and validate the specific microcircuit motifs responsible for unmasking prediction errors.

## Phase 1: Tracing the Expectation
**1. "Ghost" Trace Quantization:** 
Extract the virtual LFP and Pyramidal cell traces specifically from the V1 L2/3 population during the critical **1031ms to 1562ms** omission window ($P2$ omit) in the **AXAB** condition.
- *Target Measurement:* Establish the baseline magnitude, temporal profile, and spectral characteristics of the pure top-down expectation signal in the absence of bottom-up drive.

**2. Prediction Error (PE) Emergence:** 
Directly contrast the Pyramidal cell activity in the $P2$ window between the **AAAB** (Predicted, Zero Error) condition and the **AXAB** (Omitted, Positive Error/Ghost) condition.
- *Target Measurement:* Isolate the differential signal representing the pure Negative Prediction Error (nPE).

## Phase 2: Testing Circuit Hypotheses (Tasks 1-3)
**3. SST-Dendritic "Subtraction" Failure (Task 2):** 
Simulate the **RRRR vs. RXRR** conditions. Selectively "lesion" or suppress the $SST \to PYR$ dendritic inhibition during the $P2$ window.
- *Hypothesis Testing:* If top-down predictions reach V1 from PFC, but the SST-mediated subtractive inhibition is missing, the prediction should fail to be canceled, resulting in a massive, pathological "Ghost Signal" (false PE). Use AGSDR to find the specific $g_{SST}$ value that minimizes the difference between this simulated error and empirical "Neural Surprise."

**4. VIP-Trigger Activation (Task 3):** 
Test the VIP disinhibition hypothesis. Inject a precisely timed bias current into V1 VIP+ cells exactly during the $1031ms$ window.
- *Hypothesis Testing:* Does this injection effectively inhibit SST cells, thereby "unmasking" the top-down prediction error? If the VIP->SST connection is computationally severed, does the Omission-evoked response disappear, identifying VIP cells as the definitive "Omission Triggers"?

**5. Beta-Omission Link (Task 8):** 
Monitor the 13-30 Hz (Beta) band during the $D1 \to P2$ transition phase.
- *Hypothesis Testing:* Since feedback (prediction) is carried by Beta frequencies, verify if NMDA hypofunction (implemented in Plan 2) specifically blunts the build-up of the "expectation" Beta signal *before* the omission physically occurs.
