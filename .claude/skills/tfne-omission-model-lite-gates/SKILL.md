# Skill: TFNE Omission Model-Lite Gates

**Purpose:** Guide next omission/model-lite work without overclaiming mechanism.

**Use when:** Editing `src/jbiophysic/models/omission_lite.py`, `src/jbiophysic/models/laminar_oddball.py`, or omission tutorials.

## Required Condition Matrix
- **Baseline**
- **Unexpected Sensory**
- **Predicted Standard**
- **Omission**
- **Post-Omission** (when present)

## Required Analysis Window
- Preserve -500 ms to +1000 ms peri-event window.
- Do NOT crop at t=0.

## Required Gates
- **Omission Input:** Omission condition must have no bottom-up sensory input at the expected event time.
- **Prediction State:** Top-down/prediction state must be explicitly declared.
- **Separate Analysis:** Sparse spiking and field effects must be analyzed separately.
- **Anti-Seizure Metric:** Note synchrony anti-seizure metrics/regularizers during optimization.
- **No Mechanism-Proof:** Do not use biological mechanism-proof language.

## Validation
- Targeted condition-timing tests.
- No NaN/Inf in model outputs.
- Full test suite if source code was changed.
