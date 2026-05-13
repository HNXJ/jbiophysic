# JTFNE spectrolaminar API alignment note

Status: `truth_safe_unverified` developmental demo.

This bundle adds a notebook-facing `jtfne` API:

```python
from jbiophysic import jtfne
cfg = jtfne.default_cfg("correct", smoke=True)
jtfne.save_cfg(cfg, "cfg.yaml")
cfg = jtfne.load_cfg("cfg.yaml")
tfne_model = jtfne.construct(cfg.init)
jtfne.save_model(tfne_model, "model.pkl")
tfne_model = jtfne.load_model("model.pkl")
tfne_signals = jtfne.simulate(tfne_model, cfg.sim)
jtfne.visualize(tfne_signals, cfg.vis)
tfne_eval = jtfne.evaluate(tfne_signals, cfg.opt)
tfne_model_optimized = jtfne.optimize(tfne_model, cfg.opt)
```

## Method alignment

The implementation follows the reduced TFNE forward stack:

```text
Emitter -> Synaptic/recurrent state -> calibrated source projection -> tensor field -> probe/readout
```

The extended method PDF defines the complete forward operator as `P o F o Q o C o S o E` and optional optimizer feedback. The current code implements the source-field-probe subset plus a synaptic/recurrent Izhikevich scaffold. Chemical modulation remains a future explicit operator.

## Physical gates implemented

- Izhikevich drive is native/current-like until converted by `source_scale_A_per_native`.
- TFNE projection uses source-sink return current kernels, not unbalanced monopoles.
- Neumann compatibility is checked by `basis_conservation_max_abs`.
- Field readout metadata records gauge, boundary condition, source projection class, and solver residual proxy.
- Evaluation includes firing-rate, silent-fraction, voltage-range, and synchrony-kappa sanity diagnostics.

## Correct vs inverse laminar E/I ratio

Two modes are provided:

- `correct`: higher E/I in L5/L6 deep layers and higher inhibitory fraction in superficial layers.
- `inverse`: higher E/I in superficial layers and lower E/I in deep layers.

The scaffold-level hypothesis test is whether the `correct` laminar E/I profile scores higher against the declared target motif: deep alpha/beta and superficial gamma. This is a readout-level convergence demonstration, not proof of a biological mechanism.
