# jbiophysic

Experimental computational neuroscience framework for:

- Izhikevich and HH-style neuron models
- laminar E/PV/SST/VIP cortical circuits
- multi-area low/mid/high cortical hierarchy simulations
- global oddball and omission task scaffolds
- TFNE forward-field CSD/LFP modeling
- optimization and plasticity experiments

## Safety/status

This is exploratory research infrastructure, not a validated biological simulator. Optimizer success is not biological proof.

## Install

Minimal:

```bash
pip install -e .
```

Development:

```bash
pip install -e ".[dev]"
```

Full scientific/tutorial stack:

```bash
pip install -e ".[jax,jaxley,viz,tutorials,dev]"
```

## Quick smoke

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=src python3 -m pytest -q
```

## Tutorials

- `tutorials/00_neuronal_equations_book.ipynb`
- `tutorials/01_izhikevich_hh_single_neurons.ipynb`
- `tutorials/02_tfne_forward_fields.ipynb`
- `tutorials/03_tfne_izhikevich_hybrid.ipynb`
- `tutorials/04_laminar_oddball_three_area_cortex.ipynb`

HTML exports live in `tutorials/html/`.
