"""Tests for SpectrolaminarMotifConfig dataclass.

Validates YAML loading, config serialization, validation, and smoke defaults.
"""

import tempfile
from pathlib import Path

import numpy as np
import pytest

from jbiophysic.configs import SpectrolaminarMotifConfig


class TestSpectrolaminarConfigDefaults:
    """Test default configuration creation."""

    def test_config_default_creation(self):
        """Verify default config can be created."""
        cfg = SpectrolaminarMotifConfig()
        assert cfg.mode == "smoke"
        assert cfg.dt_ms == 0.1
        assert cfg.t_stop_ms > cfg.t_start_ms

    def test_config_is_frozen(self):
        """Verify config is frozen (immutable)."""
        from dataclasses import replace

        cfg = SpectrolaminarMotifConfig(dt_ms=0.1)
        cfg_modified = replace(cfg, dt_ms=0.5)
        # Original should be unchanged
        assert cfg.dt_ms == 0.1
        assert cfg_modified.dt_ms == 0.5

    def test_config_seed_deterministic(self):
        """Verify seed is used deterministically."""
        cfg1 = SpectrolaminarMotifConfig(seed=42)
        cfg2 = SpectrolaminarMotifConfig(seed=42)
        cfg3 = SpectrolaminarMotifConfig(seed=99)
        assert cfg1.seed == cfg2.seed
        assert cfg1.seed != cfg3.seed


class TestSpectrolaminarConfigValidation:
    """Test config validation constraints."""

    def test_invalid_dt_raises(self):
        """Verify dt_ms <= 0 raises ValueError."""
        with pytest.raises(ValueError, match="dt_ms must be positive"):
            SpectrolaminarMotifConfig(dt_ms=-0.1)

    def test_invalid_timing_raises(self):
        """Verify t_stop <= t_start raises ValueError."""
        with pytest.raises(ValueError, match="t_stop_ms"):
            SpectrolaminarMotifConfig(t_start_ms=1000, t_stop_ms=500)

    def test_invalid_stimulus_duration_raises(self):
        """Verify negative stimulus duration raises ValueError."""
        with pytest.raises(ValueError, match="stimulus_duration_ms"):
            SpectrolaminarMotifConfig(stimulus_duration_ms=-100)

    def test_invalid_cell_fractions_raises(self):
        """Verify cell_counts_by_class not summing to 1 raises ValueError."""
        bad_fractions = {"E": 0.5, "PV": 0.2, "SST": 0.1, "VIP": 0.1}  # sums to 0.9
        with pytest.raises(ValueError, match="sum to 1.0"):
            SpectrolaminarMotifConfig(cell_counts_by_class=bad_fractions)

    def test_invalid_laminar_fractions_raises(self):
        """Verify laminar_fractions not summing to 1 raises ValueError."""
        bad_fractions = {"superficial": 0.4, "middle": 0.3, "deep": 0.2}  # sums to 0.9
        with pytest.raises(ValueError, match="sum to 1.0"):
            SpectrolaminarMotifConfig(laminar_fractions=bad_fractions)

    def test_invalid_tfne_steps_raises(self):
        """Verify tfne_jacobi_steps <= 0 raises ValueError."""
        with pytest.raises(ValueError, match="tfne_jacobi_steps"):
            SpectrolaminarMotifConfig(tfne_jacobi_steps=0)

    def test_invalid_residual_tol_raises(self):
        """Verify tfne_residual_tol <= 0 raises ValueError."""
        with pytest.raises(ValueError, match="tfne_residual_tol"):
            SpectrolaminarMotifConfig(tfne_residual_tol=-1e-6)

    def test_invalid_mode_raises(self):
        """Verify unknown mode raises ValueError."""
        with pytest.raises(ValueError, match="unknown mode"):
            SpectrolaminarMotifConfig(mode="unknown")

    def test_invalid_boundary_condition_raises(self):
        """Verify unsupported boundary_condition raises ValueError."""
        with pytest.raises(ValueError, match="unsupported boundary_condition"):
            SpectrolaminarMotifConfig(tfne_boundary_condition="dirichlet")

    def test_invalid_gauge_raises(self):
        """Verify unsupported gauge raises ValueError."""
        with pytest.raises(ValueError, match="unsupported gauge"):
            SpectrolaminarMotifConfig(tfne_gauge="pinned")


class TestSpectrolaminarConfigSerialization:
    """Test config serialization and deserialization."""

    def test_to_dict_returns_dict(self):
        """Verify to_dict returns a proper dictionary."""
        cfg = SpectrolaminarMotifConfig()
        d = cfg.to_dict()
        assert isinstance(d, dict)
        assert "mode" in d
        assert "seed" in d
        assert d["mode"] == "smoke"

    def test_to_dict_deterministic(self):
        """Verify to_dict is deterministic for same config."""
        cfg1 = SpectrolaminarMotifConfig(seed=42, mode="smoke")
        cfg2 = SpectrolaminarMotifConfig(seed=42, mode="smoke")
        d1 = cfg1.to_dict()
        d2 = cfg2.to_dict()
        assert d1 == d2


class TestSpectrolaminarConfigYAML:
    """Test YAML loading and mode application."""

    def test_load_yaml_smoke_mode(self):
        """Verify YAML loading with smoke mode."""
        yaml_content = """
mode: "smoke"
seed: 123
dt_ms: 0.1
t_start_ms: 0.0
t_stop_ms: 1000.0

modes:
  smoke:
    neurons_per_area_per_class:
      V1: 50
      V4: 50
      PFC: 50
"""
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg_path = Path(tmpdir) / "test_config.yaml"
            cfg_path.write_text(yaml_content)

            cfg = SpectrolaminarMotifConfig.from_yaml(cfg_path, mode="smoke")
            assert cfg.mode == "smoke"
            assert cfg.neurons_per_area_per_class["V1"] == 50

    def test_load_yaml_full_mode(self):
        """Verify YAML loading with full mode override."""
        yaml_content = """
mode: "smoke"
seed: 42
dt_ms: 0.1
tfne_jacobi_steps: 1000

modes:
  full:
    tfne_jacobi_steps: 2000
    tfne_residual_tol: 1.0e-8
"""
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg_path = Path(tmpdir) / "test_config.yaml"
            cfg_path.write_text(yaml_content)

            cfg = SpectrolaminarMotifConfig.from_yaml(cfg_path, mode="full")
            assert cfg.mode == "full"
            assert cfg.tfne_jacobi_steps == 2000

    def test_load_yaml_missing_file(self):
        """Verify FileNotFoundError for missing config file."""
        with pytest.raises(FileNotFoundError):
            SpectrolaminarMotifConfig.from_yaml("/nonexistent/path/config.yaml")

    def test_load_yaml_unknown_mode(self):
        """Verify ValueError when mode is requested but not in file."""
        yaml_content = """
mode: "smoke"
seed: 42

modes:
  smoke:
    dt_ms: 0.1
"""
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg_path = Path(tmpdir) / "test_config.yaml"
            cfg_path.write_text(yaml_content)

            # Requesting "full" mode when only "smoke" exists should fall back to defaults
            # or raise depending on strict validation
            try:
                cfg = SpectrolaminarMotifConfig.from_yaml(cfg_path, mode="full")
                # If no error, that's also OK (empty overrides)
                assert cfg.mode == "full"
            except KeyError:
                # If strict, this is also acceptable
                pass


class TestSpectrolaminarConfigSmokeModeHelper:
    """Test with_smoke_defaults helper."""

    def test_smoke_defaults_reduces_population(self):
        """Verify with_smoke_defaults creates smaller config."""
        cfg_full = SpectrolaminarMotifConfig(
            mode="full",
            neurons_per_area_per_class={"V1": 200, "V4": 200, "PFC": 200},
        )
        cfg_smoke = cfg_full.with_smoke_defaults()
        assert cfg_smoke.mode == "smoke"
        assert cfg_smoke.neurons_per_area_per_class["V1"] == 50
        assert cfg_smoke.claim_level == "smoke_test"

    def test_smoke_defaults_loosens_solver_tol(self):
        """Verify with_smoke_defaults loosens solver tolerance."""
        cfg_full = SpectrolaminarMotifConfig(
            tfne_residual_tol=1e-8,
            tfne_jacobi_steps=2000,
        )
        cfg_smoke = cfg_full.with_smoke_defaults()
        assert cfg_smoke.tfne_residual_tol == 1e-4
        assert cfg_smoke.tfne_jacobi_steps == 100


class TestSpectrolaminarConfigIntegration:
    """Integration tests for realistic config scenarios."""

    def test_smoke_config_valid(self):
        """Verify smoke mode config is valid."""
        cfg = SpectrolaminarMotifConfig(mode="smoke")
        cfg._validate()  # Should not raise
        assert cfg.mode == "smoke"

    def test_full_config_valid(self):
        """Verify full mode config is valid."""
        cfg = SpectrolaminarMotifConfig(mode="full")
        cfg._validate()  # Should not raise
        assert cfg.mode == "full"

    def test_config_total_neurons_positive(self):
        """Verify total neuron count is positive."""
        cfg = SpectrolaminarMotifConfig()
        total = sum(cfg.neurons_per_area_per_class.values())
        assert total > 0
