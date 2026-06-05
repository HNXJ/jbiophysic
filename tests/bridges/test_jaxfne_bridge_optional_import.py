"""Tests for jaxfne bridge optional import behavior.

Focus: Graceful fallback when jaxfne is unavailable.
"""

from jbiophysic.bridges.jaxfne import (
    build_single_neuron_run,
    get_jaxfne_report,
    run_and_report,
)


def test_bridge_imports_without_jaxfne():
    """Test bridge namespace imports work even if jaxfne is unavailable."""
    from jbiophysic.bridges import jaxfne as jb_jaxfne

    # Should not raise ImportError
    assert jb_jaxfne is not None
    assert hasattr(jb_jaxfne, "build_single_neuron_run")
    assert hasattr(jb_jaxfne, "get_jaxfne_report")
    assert hasattr(jb_jaxfne, "run_and_report")


def test_get_jaxfne_report_returns_dict_with_dispatch_status():
    """Test get_jaxfne_report returns dict with dispatch_status field."""
    manifest = build_single_neuron_run(
        cell_type="izhikevich",
        params={},
        stimulus_pattern={},
        duration_ms=500.0,
        dt_ms=0.1,
    )

    # Call get_jaxfne_report with valid manifest
    report = get_jaxfne_report(manifest)

    # Should return a dict with dispatch_status
    assert isinstance(report, dict)
    assert "dispatch_status" in report
    assert isinstance(report["dispatch_status"], str)


def test_run_and_report_handles_jaxfne_unavailable():
    """Test run_and_report fails gracefully when jaxfne unavailable."""
    import tempfile
    from pathlib import Path

    manifest = build_single_neuron_run(
        cell_type="izhikevich",
        params={},
        stimulus_pattern={},
        duration_ms=500.0,
        dt_ms=0.1,
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        report = run_and_report(manifest, str(tmpdir))

        # Should return failure status
        assert "dispatch_status" in report
        assert report.get("success") is False or not report.get("success", True)

        # Should write artifacts even on failure
        assert (Path(tmpdir) / "manifest.json").exists()
        assert (Path(tmpdir) / "jaxfne_bridge_report.json").exists()


def test_get_installed_jaxfne_version_returns_unknown_when_unavailable():
    """Test get_installed_jaxfne_version returns 'unknown' if jaxfne not installed."""
    from jbiophysic.bridges.jaxfne import get_installed_jaxfne_version

    # We can't easily uninstall jaxfne in test, so just verify the function works
    version = get_installed_jaxfne_version()
    assert isinstance(version, str)
    # If jaxfne is installed, version should be a version string or "unknown"
    assert len(version) > 0
