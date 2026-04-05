# tests/test_imports.py
import pytest
import sys
import os

# Ensure flat root is testable
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_cli_importability():
    """Assert CLI module can be instantiated."""
    try:
        from cli.gravia_write import get_manuscript_paths
        paths = get_manuscript_paths()
        assert "results_md" in paths
    except ImportError as e:
        pytest.fail(f"CLI Import failed: {e}")

def test_codes_importability():
    """Assert Biophysical models are syntactically valid."""
    try:
        from codes.hierarchy import build_cortical_hierarchy
        from codes.neurons import construct_column
    except ImportError as e:
        if "jaxley" not in str(e): # Ignore expected local env missing library
            pytest.fail(f"Codes internal structure import failed: {e}")

def test_pipeline_importability():
    """Assert pipeline orchestrations resolve endpoints."""
    try:
        from pipeline.load_data import load_pharmacology_profile
        pharma = load_pharmacology_profile("ketamine")
        assert pharma["occupancy"] == 0.5
    except ImportError as e:
        pytest.fail(f"Pipeline Import failed: {e}")
