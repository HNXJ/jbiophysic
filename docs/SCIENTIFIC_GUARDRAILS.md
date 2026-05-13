# Scientific Guardrails

This document outlines the scope and limitations of the `jbiophysic` package.

- **TFNE Status**: The Total Field Neural Electrodynamics (TFNE) modules are forward models for estimating CSD and LFP-like fields. They have not been validated against whole-brain ground truth and should be treated as exploratory.
- **Izhikevich Units**: All Izhikevich model currents (`I`) are in phenomenological "model units". They do not correspond to physical amperes or nanoamperes unless a specific calibration bridge is implemented and documented.
- **Lichtenfeld Priors**: The laminar density values provided in the `laminar_oddball` module are Lichtenfeld-inspired *priors*. They are approximate three-bin summaries, not exact digitized histology.
- **Cell Type Mapping**: The mapping of markers (e.g., CB -> SST, CR -> VIP) are putative modeling choices based on common literature proxies. They are not biological equivalences.
- **Task Scaffolds**: The global oddball and omission models are task *scaffolds* designed for tutorial purposes. Their success in mimicking certain response motifs does not constitute biological proof of any specific mechanism.
- **Optimizers**: Optimization results (e.g., using Optax or custom GSDR) demonstrate mathematical reachability within the model space, not biological evidence.
