---
name: science-jbiophysics-omission-paradigm
description: Scientific implementation and hierarchical calibration of the cortical omission paradigm.
---
# science-jbiophysics-omission-paradigm

This skill details the scientific implementation and hierarchical calibration of the prediction-error paradigm.

## 1. Hierarchy & Canonical Motifs
- **Tiered Tiers**: Sensory (V1/V2), Mid-Order (V4/MT), and Executive (FEF/PFC).
- **Feedforward (FF)**: L2/3 Pyr → Higher Area L4. Mediates Gamma oscillations.
- **Feedback (FB)**: Higher Area L5/6 → Lower Area L2/3. Mediates Alpha/Beta oscillations (Top-Down Prediction).
- **Inhibitory Motifs**:
    - **PV+**: Perisomatic; Gain and Gamma control.
    - **SST+**: Dendritic; Subtractive inhibition (Prediction cancellation).
    - **VIP+**: Disinhibitory; Omission Trigger (Inhibits SST to disinhibit Pyramidal dendrites).

## 2. Four-Context Calibration Strategy
To ensure a robust model, calibration MUST occur across multiple conditions simultaneously:
1. **Context 0 (Sensory)**: BU=ON, TD=OFF. Tune FF synapses.
2. **Context 1 (Spontaneous)**: BU=OFF, TD=OFF. Tune local noise and leak.
3. **Context 2 (Predicted)**: BU=ON, TD=ON. Match oscillatory coherence and gain.
4. **Context 3 (OMISSION)**: BU=OFF, TD=ON. Match prediction-error (Ghost Signal) sparse firing.

## 3. Highly Useful Hints for Calibration
- **Temporal Alignment**: $t=0$ always at Code 101.0 (First sensory onset).
- **Ghost Signal Tuning**: If the "Ghost Signal" is too weak, increase **PFC Feedback (NMDA)** or decrease **V1 SST inhibition** (pathology mimicking).
- **Gamma/Beta Ratios**: Sensory stimuli should drive high Gamma-to-Beta ratios in V1; Omissions should reverse this to high Beta-to-Gamma ratios.
- **Kappa Metrics**: Target **Kappa < 0.10** for asynchrony. Hypersynchrony (Kappa > 0.3) often masks true prediction-error signals.

## 4. Advanced Scientific Workflows
- **Pathology Mapping**: Inducing E/I deficits (e.g., -20% PV/SST) and mapping the 3D "Hallucinogenic Valley" (Deficit Severity vs. Ghost Magnitude).
- **Spectral Match (SSS)**: MSE on log-scale PSD. Focus on Beta (13-30Hz) for feedback and Gamma (30-80Hz) for feedforward.
- **Sensitivity Knobs**: Using `jax.grad` to find which inter-areal synapses most strongly control the V1 omission response.
