# 🧬 JBiophysics: Multi-Area Cortical Research Platform

[![Architecture: Axis 1-10](https://img.shields.io/badge/Architecture-Axis%201--10-CFB87C?style=for-the-badge&logo=biophysics&logoColor=9400D3)](file:///Users/hamednejat/.gemini/antigravity/brain/dd42fe4d-98fb-42d4-8cdf-f519161852f3/walkthrough.md)

**JBiophysics** is a differentiable biophysical simulation and analysis suite built on JAX and Jaxley. It is designed to bridge the gap between high-fidelity neural circuit modeling and production-quality research posters/manuscripts.

## 🧠 Core Architecture (The Ten Axes)

| 🧬 Axis | Feature | Description |
| :--- | :--- | :--- |
| **1-4** | **Biophysics Meta-Layer** | Spiking NMDA/GABA kinetics + Hierarchical Gamma Trace (O(1) memory). |
| **5-6** | **Learning Hierarchy** | 11-area cortical hierarchy (V1 → PFC) with STDP sequence training. |
| **7-8** | **LFP Production Pipeline** | 15-step analysis (TFR, Coherence) → Poster-grade Golden Dark Figures. |
| **9-10**| **Research Automation** | `gravia-write` CLI integration + LaTeX manuscript generation. |

---

## 🚀 Getting Started

### 📦 Installation
```bash
# Clone the repository
git clone https://github.com/hnxj/jbiophysics.git
cd jbiophysics

# Install with development dependencies
uv pip install -e ".[dev]"
```

### ⌨️ CLI: `gravia-write`
The platform core is exposed through a research-grade CLI:

```bash
# 1. Run the full training-to-omission pipeline
gravia-write run --steps 5000

# 2. Render all 10 poster panels (TFR, Band Power, etc.)
gravia-write render

# 3. Export the research results to a LaTeX snippet
gravia-write write
```

---

## 📊 Poster Aesthetic
All visualizations follow the **Madelane Golden Dark** standard:
- **Base**: ![#0D0D0F](https://via.placeholder.com/15/0D0D0F/000000?text=+) `#0D0D0F`
- **Primary**: ![#CFB87C](https://via.placeholder.com/15/CFB87C/000000?text=+) `#CFB87C` (Gold)
- **Secondary**: ![#9400D3](https://via.placeholder.com/15/9400D3/000000?text=+) `#9400D3` (Violet)

---

## 📝 License
Proprietary research platform. (c) 2026 Hamed Nejat. 
Part of the **Gamma Protocol** ecosystem.
