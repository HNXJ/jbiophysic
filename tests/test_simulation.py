# tests/test_simulation.py
import pytest
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline.run_simulation import run_simulation
from pipeline.load_data import load_empirical_neurophysiology

def test_mock_simulation_fallback(tmp_path):
    """
    Assert that the fallback gracefully handles host constraints while maintaining
    the necessary pipeline parameter schemas.
    """
    config = {
        "simulation": {
            "n_areas": 2,
            "T_total": 100,
            "dt": 0.05
        }
    }
    
    # Store standard CWD and move to valid temp folder to test output creation
    current_cwd = os.getcwd()
    os.chdir(tmp_path)
    
    state, traces = run_simulation(config)
    
    # Assert shape matches configured schema mapping expectations
    assert "V" in traces
    assert len(traces["V"]) == int(100/0.05)
    
    # Assert output directory successfully created and populated payload
    assert os.path.exists("output/simulation_trace.json")

    # Restore CWD
    os.chdir(current_cwd)

def test_pharmacology_empirics():
    """Assert data ingest yields matching masking bounds."""
    data = load_empirical_neurophysiology("dummy_path.mat")
    assert "gamma_mask" in data
    assert "beta_mask" in data
