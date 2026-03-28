---
name: science-jbiophysics-omission-paradigm
description: Scientific implementation and hierarchical calibration of the cortical omission paradigm in jbiophysics.
---
# science-jbiophysics-omission-paradigm

This skill details the scientific implementation and hierarchical calibration of the prediction paradigm. It models the observation that cortical columns maintain internal predictions, and sensory absence triggers a top-down modulated "omission" state.

## 1. Hierarchy & Canonical Motifs
- **V1 Column (200 neurons)**: Three-layered (L2/3, L4, L5/6). Focus on PV/SST/VIP inhibitory microcircuits.
- **HO Column (100 neurons)**: Simplified higher-order column providing top-down predictions.
- **Feedforward (FF)**: V1 L2/3 → HO L4 (Gamma).
- **Feedback (FB)**: HO L5/6 → V1 L2/3 (Alpha/Beta).

## 2. Four-Context Calibration
Calibration using `OptimizerFacade` is performed across multiple trial states:
1. **Context 0 (BU=ON, TD=OFF)**: Sensory baseline. Tune `gAMPA` for sensory drive.
2. **Context 1 (BU=OFF, TD=OFF)**: Spontaneous baseline. Tune `Inoise` and `gLeak` for 1-5Hz firing.
3. **Context 2 (BU=ON, TD=ON)**: Predicted stimulus. Match oscillatory coherence.
4. **Context 3 (BU=OFF, TD=ON)**: **OMISSION**. Match sparse V1 firing and HO alpha/beta modulation.

## 3. Hierarchical Calibration Protocol
Calibration is performed in two stages:
1. **Isolated Stage**: Tune each area independently to achieve local EI balance (`Kappa < 0.1`).
2. **Joint Stage**: Enable FF/FB connections and use area-specific constraints to ensure stability during inter-areal driving.

### Multi-Objective Loss (Weights)
- **Global FR** (1.0): Network-wide stability.
- **Population FR** (5.0): Laminar-specific physiological targets.
- **Synchrony (Kappa)** (100.0): Strong penalty to prevent hypersynchrony.
- **Spectral Profile (PSD)** (10.0): Matching oscillatory peaks (e.g., V1 Alpha).

## 4. Advanced Workflows
- **Sensitivity Analysis**: Using `jax.grad` to map the influence of specific synapses on global metrics (e.g., Gamma power).
- **Curriculum Learning**: Phased optimization starting with coarse firing rates and refining towards high-resolution PSD matching.
- **Real-time Monitoring**: Streaming gradients to the API `/tuning/status` for live tracking of the optimization landscape.
