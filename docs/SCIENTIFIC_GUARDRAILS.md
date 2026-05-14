# Scientific Guardrails

This document outlines the scope and limitations of the `jbiophysic` package.

- **TFNE Status**: The Tensor-Field Neural Equations (TFNE) modules are forward models for estimating CSD and LFP-like fields. They have not been validated against whole-brain ground truth and should be treated as exploratory.
- **Izhikevich Units**: All Izhikevich model currents (`I`) are in phenomenological "model units". They do not correspond to physical amperes or nanoamperes unless a specific calibration bridge is implemented and documented.
- **Lichtenfeld Priors**: The laminar density values provided in the `laminar_oddball` module are Lichtenfeld-inspired *priors*. They are approximate three-bin summaries, not exact digitized histology.
- **Cell Type Mapping**: The mapping of markers (e.g., CB -> SST, CR -> VIP) are putative modeling choices based on common literature proxies. They are not biological equivalences.
- **Task Scaffolds**: The global oddball and omission models are task *scaffolds* designed for tutorial purposes. Their success in mimicking certain response motifs does not constitute biological proof of any specific mechanism.
- **Optimizers**: Optimization results (e.g., using Optax or custom GSDR) demonstrate mathematical reachability within the model space, not biological evidence.

## TFNE source decomposition and proxy-readout guardrails

Baseline TFNE does **not** include `q_chem` in the electrical source-density term. Chemical and neuromodulatory variables are parameter modulators unless an explicit electrodiffusive ionic-flux model is implemented.

To avoid double counting synaptic current, use either a total membrane-current source
`q = chi * I_m_total + q_ext`, where `I_m_total = I_cap + I_ion + I_syn`, or an explicitly decomposed source
`q = q_cap_ion + q_syn + q_ext`, where `q_cap_ion = chi*(I_cap + I_ion)` and `q_syn = chi*I_syn`. Do not add both notations simultaneously.

CSD convention: `CSD = div(J_e)`. Positive CSD denotes extracellular source under this convention; experimental sink plots may use the opposite sign.

JTFNE spectrolaminar outputs are proxy readouts unless exact-run physical invariants, source calibration, and empirical amplitude calibration are supplied.
