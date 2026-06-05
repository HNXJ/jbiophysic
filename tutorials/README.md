# jbiophysic tutorial notebooks

This directory contains portable source notebooks backed by the latest
`jbiophysic` package modules in `src/jbiophysic`. The repo intentionally avoids
tracking generated HTML exports, executed notebook copies, and Colab-specific
notebooks unless explicitly approved.

The notebooks are book-style, executable smoke tutorials. They are exploratory teaching
artifacts and should not be treated as validated biological claims.

## Notebooks

1. `00_neuronal_equations_book.ipynb` — equation-family overview.
2. `01_izhikevich_hh_single_neurons.ipynb` — Izhikevich and HH single-neuron tutorials.
3. `02_tfne_forward_fields.ipynb` — TFNE grid, gauge, SPD tensor, and source conservation.
4. `03_tfne_izhikevich_hybrid.ipynb` — cortex builder + calibrated Izhikevich-to-TFNE source projection.
5. `04_laminar_oddball_three_area_cortex.ipynb` — compact three-area laminar oddball/omission scaffold.

Generated HTML/executed copies should be written outside tracked source paths or regenerated locally as needed.

## Run

From the repository root:

```bash
PYTHONPATH=src jupyter nbconvert --to notebook --execute tutorials/00_neuronal_equations_book.ipynb --inplace
PYTHONPATH=src jupyter nbconvert --to notebook --execute tutorials/01_izhikevich_hh_single_neurons.ipynb --inplace
PYTHONPATH=src jupyter nbconvert --to notebook --execute tutorials/02_tfne_forward_fields.ipynb --inplace
PYTHONPATH=src jupyter nbconvert --to notebook --execute tutorials/03_tfne_izhikevich_hybrid.ipynb --inplace
PYTHONPATH=src jupyter nbconvert --to notebook --execute tutorials/04_laminar_oddball_three_area_cortex.ipynb --inplace
```

## Scientific guardrails

- Izhikevich current is native current-like drive, not nA.
- TFNE uses SI units and explicit calibration before current-like point-neuron events become sources.
- Lichtenfeld-inspired density priors are three-layer tutorial summaries, not raw digitized histology.
- Westerberg-style global oddball and omission tasks share timing; only condition sequence changes.
- Passing smoke tests means the scaffold is executable, not that the papers have been replicated.

## Improved replication scaffold additions

The three-area tutorial now includes:

- `tutorials/data/lichtenfeld_three_layer_priors.csv`: machine-readable three-layer density priors.
- `tutorials/data/three_area_replication_manifest.json`: default 300-neuron scaffold manifest.
- validation audit via `validate_replication_constraints`.
- perturbation handles via `edge_mask` and `perturb_cortex_edges`.
- batch global-oddball and omission simulation helpers.
- compact objective functions for local/global oddball and omission contrasts.
- population activity proxies and a TFNE sparse-source bridge.

These helpers make it easier to move from tutorial execution to paper-constrained model fitting without hiding assumptions in notebook cells.
