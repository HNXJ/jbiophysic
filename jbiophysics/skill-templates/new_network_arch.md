# Adding a New Network Architecture

1. Create `jbiophysics/systems/networks/your_network.py`
2. Use `from jbiophysics.core.mechanisms.models import ...` for mechanisms
3. Use `from jbiophysics.core.neurons.hh_cells import ...` for cell builders
4. Register in `jbiophysics/systems/networks/__init__.py`
5. Optionally add to `CELL_BUILDERS` or `SYNAPSE_TYPES` in `compose.py`
