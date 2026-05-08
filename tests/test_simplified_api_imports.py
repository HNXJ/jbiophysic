import pytest

def test_simplified_imports():
    from jbiophysic.neurons import IzhikevichParams
    from jbiophysic.conditions import Condition
    from jbiophysic.objectives import Objective
    from jbiophysic import ops
    from jbiophysic.viz import jvis, JVis, serialize_raster
    from jbiophysic.fields import make_regular_grid
    from jbiophysic.models import build_cortical_hierarchy, run_simulation
    
    assert IzhikevichParams is not None
    assert Condition is not None
    assert Objective is not None
    assert ops.firing_rate is not None
    assert jvis.raster is not None
    assert make_regular_grid is not None
    assert build_cortical_hierarchy is not None
    assert run_simulation is not None
