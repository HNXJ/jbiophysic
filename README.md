# 🧬 jbiophys

### A Differentiable Framework for Multi-Area Cortical Dynamics, Predictive Coding, and Synaptic Optimization

---

## 📖 Overview

**jbiophys** is a computational neuroscience framework for simulating, optimizing, and analyzing **multi-area cortical dynamics** using biologically grounded mechanisms and differentiable simulation.

The framework integrates:

* Conductance-based neural dynamics (AMPA, NMDA, GABA)
* Cell-type specific microcircuits (Pyramidal, PV, SST, VIP)
* Hierarchical cortical organization (V1 → PFC)
* Predictive coding architectures
* Oscillatory dynamics (gamma: feedforward, beta: feedback)
* Synaptic plasticity and optimization (GSDR / AGSDR / GSGD)

It provides a unified pipeline for:

```text
simulation → optimization → LFP analysis → figures → manuscript
```

---

## 🧠 Scientific Scope

This framework is designed to study:

* Predictive processing in cortical hierarchies
* Oscillatory coordination across brain regions
* Excitation–Inhibition balance and stability
* Emergence of beta/gamma dynamics
* Neural responses to structured perturbations (e.g., omission paradigms)

---

## ⚙️ Core Components

### 1. Multi-Area Cortical Hierarchy
* Hierarchical structure spanning sensory (V1) to prefrontal (PFC) cortex.
* Bidirectional connectivity: Feedforward (gamma-mediated) and Feedback (beta-mediated).

### 2. Mechanism-Based Modeling
All dynamics are defined through composable mechanisms:
* Ion channels (Hodgkin–Huxley formalism)
* Synapses (conductance-based AMPA, NMDA, GABA-A/B kinetics)
* Neuromodulators (Dopamine, Acetylcholine)
* Plasticity rules (STDP and extensions)

### 3. Oscillatory Dynamics
The framework captures:
* **Gamma-band (~30–80 Hz)**: Feedforward signaling.
* **Beta-band (~13–30 Hz)**: Feedback / predictive coordination.
* Emergent from E/I balance (PV, SST circuits).

### 4. LFP Analysis Pipeline
A standardized multi-step pipeline providing TFR, spectral bands, coherence, and Granger causality.

---

## 🧪 Synaptic Optimization Framework

### 🔁 GSDR — Genetic Synaptic Drift Rule
A population-based optimization framework for exploring synaptic parameter space through mutation, crossover, and selection.

### 🧠 AGSDR — Adaptive Gradient Synaptic Drift Regularization
A gradient-based optimization layer enforcing physiological constraints (rate, E/I balance, stability) via:
$w_{t+1} = \text{clip}(w_t - \eta \cdot \text{clip}(\nabla L))$

### ⚡ GSGD — Genetic–Stochastic Gradient Descent
A hybrid optimization framework combining global genetic search (GSDR) with local stochastic gradient refinement (SGD) and AGSDR constraints.

---

## 🧠 Experimental Paradigm: Omission Task
Supports sequence learning (S1 → S2 → S3), oddball (S1 → S2 → S4), and omission (S1 → S2 → ∅).

---

## 🚀 Usage

Run the full experimental pipeline:
```bash
python codes/scripts/run_full_experiment.py
```

---

## 🔬 Design Principles
* Mechanism-first modeling
* Biologically grounded constraints
* Differentiable simulation (JAX-native)
* Scalable optimization (Multi-device GSGD ready)
