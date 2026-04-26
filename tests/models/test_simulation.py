# tests/models/test_simulation.py
import pytest
import jax.numpy as jnp
from jbiophysic.models.builders.hierarchy import build_cortical_hierarchy
from jbiophysic.models.simulation.run import run_simulation
from jbiophysic.common.types.simulation import SimulationConfig

def test_hierarchy_build():
    """Verifies area-to-cell scaling in the hierarchical cortical builder."""
    n_areas = 2
    brain = build_cortical_hierarchy(n_areas=n_areas)
    
    assert len(brain.cells) > 0
    # Canonical construct_column: 200 PC + 40 PV + 40 SST + 20 VIP = 300 cells per area
    expected_cells = n_areas * 300
    assert len(brain.cells) == expected_cells

def test_simulation_run():
    """Verifies integration output shapes and finiteness for a short run."""
    n_areas = 1
    brain = build_cortical_hierarchy(n_areas=n_areas)
    
    config = SimulationConfig(t_max=10.0, dt=0.1)
    result = run_simulation(brain, config)
    
    assert result.v_trace.shape[0] == len(brain.cells)
    # t_max=10, dt=0.1 => 100 steps (Jaxley indexing)
    assert result.v_trace.shape[1] == 100
    assert jnp.all(jnp.isfinite(result.v_trace))
    assert result.currents is not None
