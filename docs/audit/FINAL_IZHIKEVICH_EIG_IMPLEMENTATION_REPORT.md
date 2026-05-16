# Final Izhikevich E/Ing/Inl Network Implementation Report

Status: FINAL implementation patch
Truth mode: truth_safe_unverified
Claim level: computational_scaffold
Version target: 1.0.3

## Summary

Implemented the supplied `net_eig(num_e, num_ig, num_il)` network motif with Izhikevich point-neuron emitters.

The implementation preserves the original scaffold semantics:

- `E` excitatory population.
- `Ing` global inhibitory interneurons, SST-like / low-threshold-spiking.
- `Inl` local inhibitory interneurons, PV-like / fast-spiking.
- Two logical connection sites matching the original JAXley branch/localization contract:
  - pre: branch `0`, loc `0.0`;
  - post: branch `1`, loc `1.0`.
- E -> all AMPA, `tauD = 2 ms`.
- Ing -> all GABAa, `tauD = 5 ms`.
- Inl -> selected 10% posts GABAa, `tauD = 2 ms`, using `jax.random.PRNGKey(1)` by default to match the supplied constructor logic.
- Uniform radius and length metadata equal to `1.0`.
- Inoise-like native-current noise metadata with `tau ~ U(5, 50)` and clipped amplitude from `U(-0.1, 0.1)` to `[0, 0.1]`.

## Files changed

- `src/jbiophysic/networks/izhikevich_eig.py`
- `src/jbiophysic/networks/__init__.py`
- `tests/test_izhikevich_eig_network.py`
- `pyproject.toml`
- `docs/audit/FINAL_IZHIKEVICH_EIG_IMPLEMENTATION_REPORT.md`

## Public API

```python
from jbiophysic.networks.izhikevich_eig import net_eig, simulate_eig_izhikevich

net = net_eig(num_e=80, num_ig=10, num_il=10)
```

Also exported through `jbiophysic.networks`:

```python
from jbiophysic.networks import net_eig, make_izhikevich_eig_network, simulate_eig_izhikevich
```

## Izhikevich parameter mapping

| Population | Biological role | Izhikevich regime | Parameters |
|---|---|---|---|
| `E` | excitatory | regular spiking | `a=0.02, b=0.20, c=-65, d=8` |
| `Ing` | global inhibitory / SST-like | low-threshold-spiking | `a=0.02, b=0.25, c=-65, d=2` |
| `Inl` | local inhibitory / PV-like | fast-spiking | `a=0.10, b=0.20, c=-65, d=2` |

## Claim boundary

This is an Izhikevich reduced-emitter network. Native current is not physical amperes.

The network metadata explicitly marks:

```text
source_calibration_status = uncalibrated_izhikevich_native_current
source_claim = reduced_emitter_spike_state_only_not_physical_current
```

Use the TFNE calibration bridge before making physical CSD/LFP amplitude claims.

## Validation commands

```bash
python -m compileall -q src tests
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=src python -m pytest -q tests/test_izhikevich_eig_network.py
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=src python -m pytest -q tests/test_izhikevich_eig_network.py tests/test_cortex_network_builder.py
```

## Observed validation

- Compile: passed.
- New Izhikevich EIG tests: `3 passed`.
- New tests + cortex network builder regression: `8 passed`.

## Decision

`FINAL_READY_FOR_MERGE`
