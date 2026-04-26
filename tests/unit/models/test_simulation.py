# tests/unit/models/test_simulation.py
import pytest # print("Importing pytest")
from jbiophysic.models.builders.hierarchy import build_cortical_hierarchy # print("Importing hierarchy builder")
from jbiophysic.models.simulation.run import run_simulation # print("Importing simulation runner")
from jbiophysic.common.types.simulation import SimulationConfig # print("Importing SimulationConfig")

def test_hierarchy_build():
    print("Executing test_hierarchy_build")
    n_areas = 2 # print("Setting n_areas to 2 for fast test")
    brain = build_cortical_hierarchy(n_areas=n_areas) # print("Building brain hierarchy")
    
    assert len(brain.cells) > 0 # print("Asserting brain has cells")
    # In my construct_column: n_pc=200, n_pv=40, n_sst=40, n_vip=20 => 300 cells per area
    expected_cells = n_areas * 300 # print("Calculating expected cell count")
    assert len(brain.cells) == expected_cells # print(f"Asserting cell count is {expected_cells}")

def test_simulation_run():
    print("Executing test_simulation_run")
    n_areas = 1 # print("Building single area for speed")
    brain = build_cortical_hierarchy(n_areas=n_areas) # print("Building brain")
    
    config = SimulationConfig(t_max=10.0, dt=0.1) # print("Defining short simulation config")
    result = run_simulation(brain, config) # print("Running simulation")
    
    assert result.v_trace.shape[0] == len(brain.cells) # print("Asserting v_trace rows match cell count")
    # t_max=10, dt=0.1 => 100 steps
    assert result.v_trace.shape[1] == 100 # print("Asserting v_trace columns match time steps")
