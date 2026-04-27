# 🧠 jbiophysic

`jbiophysic` is an experimental JAX/Jaxley codebase for building simple cortical cell populations, assembling small hierarchy models, running voltage-trace simulations, and extracting basic analysis features. It is currently best treated as a research prototype rather than a finished simulator.

## 🛠 Implemented today
The following features are implemented and functional in the current codebase:
- **Jaxley-based Simulations**: Multi-compartment biophysical simulations using the Jaxley 0.5.0 engine.
- **Biophysical Mechanisms**: Standard Hodgkin-Huxley channel kinetics and basic synaptic models (AMPA, GABAa, NMDA).
- **Cell Populations**: Builders for canonical cortical populations including Pyramidal Cells (PC), PV, SST, and VIP interneurons.
- **Microcircuit Assembly**: Cortical column construction with explicit population labeling and connectivity.
- **Areal Hierarchies**: Assembly of multi-area cortical networks with inter-areal feedforward and feedback routing.
- **Spectral Analysis**: Basic signal processing for extracting beta and gamma band power from simulated traces.
- **Activity Serialization**: Decoupled serialization of spike rasters and voltage traces for downstream processing.

## 🔬 Experimental
These components are present in the repository but are considered experimental or partially integrated:
- **Differentiable Optimization**: Early implementations of GSGD (Global Stochastic Gradient Descent) and AGSDR loops for parameter tuning.
- **Surrogate Plasticity**: Differentiable surrogate spike functions for JAX-native gradient flow in synaptic learning rules.
- **Predictive Coding Primitives**: Mathematical kernels for precision-weighted error propagation.

## 🚀 Planned
We are working toward the following architectural and ergonomic improvements:
- **Ergonomic API**: A consolidated scientist-facing interface (`network.simulate()`, `network.train()`) to reduce boilerplate.
- **Unified Network Abstraction**: Deeper integration between JAX/Equinox state management and the Jaxley simulation engine.
- **Validation Dashboard**: Integrated visualization tools for rapid circuit diagnostics and E/I balance auditing.
- **Biophysical Verification**: Systematic benchmarking against empirical cortical datasets.

## ⚠️ Known Limitations
- **API Volatility**: The internal interfaces between builders and simulation runners are subject to change as the 3-tier architecture matures.
- **Unit Sensitivity**: Biophysical parameters require careful coordination across layers to maintain physiological realism.
- **Visualization Decoupling**: Plotting utilities currently rely on external serializers and are not yet unified into a single dashboard.

## ⚙️ Installation
Install the package in development mode:
```bash
pip install -e ".[dev,viz]"
```

## 📜 Engineering Standards
- **JIT Purity**: Mathematical kernels are kept free of side-effects to ensure optimal JAX compilation.
- **Structured Logging**: Centralized logging is used for system transparency.
- **Root Hygiene**: All implementation logic is strictly contained within the `src/jbiophysic/` directory.
