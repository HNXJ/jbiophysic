# 🧠 jbiophysic

`jbiophysic` is an experimental JAX/Jaxley codebase for building simple cortical cell populations, assembling multi-area hierarchy models, and running biophysical simulations. It is a research prototype focusing on exploratory modeling rather than a production-ready simulator.

## 🛠 Implemented today
- **Jaxley-based Simulations**: Core simulation loop using Jaxley 0.5.0.
- **Biophysical Mechanisms**: HH channel kinetics and synaptic models (AMPA, GABAa, NMDA).
- **Cell Populations**: Builders for pyramidal cells and PV/SST/VIP interneuron types.
- **Microcircuit Assembly**: Cortical column construction with explicit population labeling.
- **Areal Hierarchies**: Assembly of networks with feedforward and feedback routing.
- **Spectral Analysis**: Basic signal analysis for beta and gamma band power.

## 🔬 Experimental
- **Optimization Loops**: Prototype implementations of GSGD and AGSDR engines.
- **Surrogate Plasticity**: Differentiable surrogate spikes for JAX gradient flow.
- **Predictive Coding**: Preliminary mathematical kernels for hierarchical error propagation.

## 🚀 Planned
- **Ergonomic API**: Unified `network.simulate()` and `network.train()` interfaces.
- **Validation Suite**: Systematic benchmarking against empirical biophysical datasets.
- **Integrated Dashboard**: Unified visualization for circuit diagnostics.

## ⚠️ Known Limitations
- **Prototype Status**: Internal APIs are subject to change as the architecture matures.
- **Simulation Scope**: Currently optimized for exploratory small-to-mid-scale hierarchies.

## ⚙️ Installation
```bash
pip install -e ".[dev,viz]"
```

## 📜 Engineering Standards
- **JIT Purity**: Mathematical kernels are side-effect free for JAX compatibility.
- **Root Hygiene**: Implementation resides strictly within `src/jbiophysic/`.
