# 🧬 jbiophys

**A Differentiable Biophysical Framework for Multi-Area Cortical Dynamics, Predictive Coding, and Oscillatory Neuroscience**

---

## 📖 Overview

**jbiophys** is a computational neuroscience framework for simulating and analyzing **multi-area cortical dynamics** using biologically grounded mechanisms, including:

* Conductance-based synapses (AMPA, NMDA, GABA-A/B)
* Cell-type specific microcircuits (Pyramidal, PV, SST, VIP, etc.)
* Predictive coding architectures across cortical hierarchies (V1 → PFC)
* Neuromodulatory control (Dopamine, Acetylcholine)
* Synaptic plasticity (STDP and extensions)

The system is designed to support **end-to-end scientific workflows**:

```text
simulation → LFP analysis → figure generation → manuscript
```

---

## 🧠 Scientific Motivation

Cortical computation is shaped by:

* Hierarchical predictive processing
* Oscillatory coordination (gamma: feedforward, beta: feedback)
* Sparse, distributed coding
* Neuromodulatory regulation of gain and precision

This repository provides a unified framework to:

* Simulate these processes from first principles
* Reproduce electrophysiological findings (e.g., omission paradigms)
* Generate publication-ready analyses and figures

---

## ⚙️ Core Features

### 1. Multi-Area Cortical Hierarchy
* Hierarchical organization (V1 → V2 → V4 → MT → … → PFC)
* Bidirectional connectivity (feedforward / feedback)
* Area-specific timescales

### 2. Mechanism-Based Modeling
All dynamics are defined through composable mechanisms:
* Ion channels (Hodgkin–Huxley style)
* Synapses (conductance-based)
* Neuromodulators (parameter modulation)
* Plasticity rules (STDP, extensions)

### 3. Oscillatory Dynamics
* **Gamma (~30–80 Hz)**: Feedforward processing
* **Beta (~13–30 Hz)**: Feedback / prediction
* Emergent from E/I balance (PV, SST circuits)

### 4. LFP Analysis Pipeline (15-step)
Includes standard spectral decomposition, coherence matrices, and cluster-based statistical comparisons.

---

## 🧪 Learning & Optimization

### 🔁 STDP (Synaptic Plasticity)
* Synapse-specific gating (`stdp_on`) and scaling (`stdp_delta`).

### 🧠 AGSDR (Adaptive Gradient Synaptic Drift Regularization)
* A pre-optimization framework used to stabilize network dynamics and enforce biologically plausible firing regimes before experimental simulation.

### ⚡ GSDR / GSGD (Planned)
* General synaptic drift regularization framework and SGD-based implementation for scalable optimization.

---

## 🚀 Usage

### Run full experiment
```bash
python codes/scripts/run_full_experiment.py
```

### Outputs
```text
output/
  lfp_signals.npy
  lfp_results.json

figures/
  (poster-style panels)

manuscript/
  main.pdf
```

---

## 🔬 Future Directions
* Calcium-dependent plasticity (NMDA-driven)
* Large-scale multi-session fitting (AGSDR)
* GPU/TPU/Apple Silicon optimization

---

## 📌 Design Philosophy
* **Mechanism-first** (not model-first)
* **Analysis-integrated simulation**
* **Reproducible by construction**
* **Agent-compatible pipelines**
