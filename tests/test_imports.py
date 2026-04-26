# tests/test_imports.py
import pytest

def test_cli_importability():
    """Validates that the CLI entry point is accessible."""
    from jbiophysic.cli.main import main
    assert main is not None

def test_core_importability():
    """Validates that the biophysical core mechanisms and math are accessible."""
    from jbiophysic.core import HH, SpikingNMDA, predictive_step
    assert HH is not None
    assert SpikingNMDA is not None

def test_models_importability():
    """Validates that the orchestration tier (builders and simulation) is accessible."""
    from jbiophysic.models import build_cortical_hierarchy, run_simulation
    assert build_cortical_hierarchy is not None
    assert run_simulation is not None

def test_viz_importability():
    """Validates that the visualization serializers are accessible."""
    from jbiophysic.viz import serialize_raster
    assert serialize_raster is not None

def test_common_importability():
    """Validates that common utilities like logging and serialization are accessible."""
    from jbiophysic.common.utils.logging import get_logger
    from jbiophysic.common.utils.serialization import safe_serialize_json
    assert get_logger is not None
    assert safe_serialize_json is not None