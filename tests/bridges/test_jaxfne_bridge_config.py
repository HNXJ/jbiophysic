"""Tests for jaxfne bridge config dataclasses.

Focus: Config validation, canonical constants, time grid validation.
"""

from jbiophysic.bridges.jaxfne import config


def test_import_works():
    """Test import path."""
    from jbiophysic.bridges import jaxfne as jb_jaxfne

    assert jb_jaxfne is not None
    assert hasattr(jb_jaxfne, "BridgeConfig")
    assert hasattr(jb_jaxfne, "SingleNeuronConfig")
    assert hasattr(jb_jaxfne, "EINetworkConfig")
    assert hasattr(jb_jaxfne, "LaminarProxyConfig")


def test_canonical_constants():
    """Test canonical literal values."""
    assert config.TRUTH_MODE == "truth_safe_unverified"
    assert config.CLAIM_LEVEL == "computational_scaffold"
    assert config.BRIDGE_VERSION == "bridges.jaxfne.v0.1"


def test_bridge_config_defaults():
    """Test base BridgeConfig defaults."""
    cfg = config.BridgeConfig()
    assert cfg.truth_mode == "truth_safe_unverified"
    assert cfg.claim_level == "computational_scaffold"
    assert cfg.jbiophysic_bridge_version == "bridges.jaxfne.v0.1"
    assert cfg.physical_amplitude_claim_allowed is False
    assert cfg.allow_nan_in_manifest is False
    assert cfg.seed == 0


def test_bridge_config_validates():
    """Test BridgeConfig validation."""
    cfg = config.BridgeConfig()
    is_valid, errors = cfg.validate()
    assert is_valid, f"Valid config should pass: {errors}"
    assert len(errors) == 0


def test_bridge_config_rejects_wrong_truth_mode():
    """Test BridgeConfig rejects wrong truth_mode."""
    cfg = config.BridgeConfig(truth_mode="wrong_mode")
    is_valid, errors = cfg.validate()
    assert not is_valid
    assert any("truth_mode" in e for e in errors)


def test_bridge_config_rejects_amplitude_claim():
    """Test BridgeConfig rejects physical_amplitude_claim_allowed=True."""
    cfg = config.BridgeConfig(physical_amplitude_claim_allowed=True)
    is_valid, errors = cfg.validate()
    assert not is_valid
    assert any("physical_amplitude_claim_allowed" in e for e in errors)


def test_single_neuron_config_defaults():
    """Test SingleNeuronConfig defaults."""
    cfg = config.SingleNeuronConfig()
    assert cfg.cell_type == "izhikevich"
    assert cfg.duration_ms == 1000.0
    assert cfg.dt_ms == 0.1


def test_single_neuron_config_validates():
    """Test SingleNeuronConfig validation."""
    cfg = config.SingleNeuronConfig(
        cell_type="izhikevich",
        params={"a": 0.02, "b": 0.2},
        duration_ms=500.0,
        dt_ms=0.1,
    )
    is_valid, errors = cfg.validate()
    assert is_valid, f"Valid config should pass: {errors}"


def test_single_neuron_config_rejects_invalid_cell_type():
    """Test SingleNeuronConfig rejects invalid cell_type."""
    cfg = config.SingleNeuronConfig(cell_type="invalid_neuron")
    is_valid, errors = cfg.validate()
    assert not is_valid
    assert any("cell_type" in e for e in errors)


def test_single_neuron_config_rejects_invalid_duration():
    """Test SingleNeuronConfig rejects invalid duration."""
    cfg = config.SingleNeuronConfig(duration_ms=-100.0)
    is_valid, errors = cfg.validate()
    assert not is_valid
    assert any("duration_ms" in e for e in errors)


def test_single_neuron_config_rejects_invalid_dt():
    """Test SingleNeuronConfig rejects invalid dt."""
    cfg = config.SingleNeuronConfig(dt_ms=0.0)
    is_valid, errors = cfg.validate()
    assert not is_valid
    assert any("dt_ms" in e for e in errors)


def test_single_neuron_config_checks_time_grid():
    """Test SingleNeuronConfig validates time grid resolution."""
    # Valid: 500 / 0.1 = 5000 steps exactly
    cfg = config.SingleNeuronConfig(duration_ms=500.0, dt_ms=0.1)
    is_valid, errors = cfg.validate()
    assert is_valid, f"Valid time grid should pass: {errors}"

    # Invalid: 500 / 0.3 = 1666.666... steps (not exact)
    cfg = config.SingleNeuronConfig(duration_ms=500.0, dt_ms=0.3)
    is_valid, errors = cfg.validate()
    assert not is_valid, "Non-exact time grid should fail"
    assert any("residual" in e or "dt_ms" in e for e in errors)


def test_ei_network_config_defaults():
    """Test EINetworkConfig defaults."""
    cfg = config.EINetworkConfig()
    assert cfg.n_exc == 100
    assert cfg.n_inh == 25
    assert cfg.duration_ms == 2000.0
    assert cfg.dt_ms == 0.1


def test_ei_network_config_validates():
    """Test EINetworkConfig validation."""
    cfg = config.EINetworkConfig(n_exc=50, n_inh=10)
    is_valid, errors = cfg.validate()
    assert is_valid, f"Valid config should pass: {errors}"


def test_ei_network_config_rejects_negative_n_exc():
    """Test EINetworkConfig rejects negative n_exc."""
    cfg = config.EINetworkConfig(n_exc=-1)
    is_valid, errors = cfg.validate()
    assert not is_valid
    assert any("n_exc" in e for e in errors)


def test_ei_network_config_rejects_zero_neurons():
    """Test EINetworkConfig rejects zero total neurons."""
    cfg = config.EINetworkConfig(n_exc=0, n_inh=0)
    is_valid, errors = cfg.validate()
    assert not is_valid
    assert any("total neurons" in e for e in errors)


def test_laminar_proxy_config_defaults():
    """Test LaminarProxyConfig defaults."""
    cfg = config.LaminarProxyConfig()
    assert cfg.source_scale == "proxy"
    assert cfg.duration_ms == 5000.0
    assert cfg.dt_ms == 0.1


def test_laminar_proxy_config_validates():
    """Test LaminarProxyConfig validation."""
    cfg = config.LaminarProxyConfig(source_scale="toy")
    is_valid, errors = cfg.validate()
    assert is_valid, f"Valid config should pass: {errors}"


def test_laminar_proxy_config_rejects_invalid_source_scale():
    """Test LaminarProxyConfig rejects invalid source_scale."""
    cfg = config.LaminarProxyConfig(source_scale="invalid_scale")
    is_valid, errors = cfg.validate()
    assert not is_valid
    assert any("source_scale" in e for e in errors)


def test_laminar_proxy_config_all_source_scales():
    """Test LaminarProxyConfig with all valid source_scales."""
    for scale in ("toy", "proxy", "calibrated", "physical"):
        cfg = config.LaminarProxyConfig(source_scale=scale)
        is_valid, errors = cfg.validate()
        assert is_valid, f"Scale {scale} should be valid: {errors}"
