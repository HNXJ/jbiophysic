from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from jbiophysic.playgrounds.jaxfne_networks import (
    JaxfnePlaygroundSpec,
    available_playgrounds,
    build_jaxfne_config,
    build_request_manifest,
    jaxfne_backend_report,
    run_playground_smoke,
    validate_signal_contract,
)


class _FakeModel:
    def select(self, *, cell_type=None):
        if cell_type == "E":
            return [0, 1]
        return [0, 1, 2, 3]


class _FakeSignals:
    def __init__(self):
        self.V_m = np.ones((5, 4), dtype=np.float32)
        self.spk = np.zeros((5, 4), dtype=np.float32)

    def get(self, name, **kwargs):
        if name == "vm" and kwargs.get("cell_type") == "E":
            return self.V_m[:, :2]
        if name == "vm":
            return self.V_m
        if name == "spk":
            return self.spk
        raise KeyError(name)


class _FakeJaxfne:
    __version__ = "0.3.fake"

    def __init__(self):
        self.calls = []

    def suite2_four_celltype_config(self, **kwargs):
        self.calls.append(("suite2_four_celltype_config", kwargs))
        return {"kind": "four", **kwargs}

    def suite2_single_neuron_config(self, **kwargs):
        self.calls.append(("suite2_single_neuron_config", kwargs))
        return {"kind": "single", **kwargs}

    def suite2_net1_config(self, **kwargs):
        self.calls.append(("suite2_net1_config", kwargs))
        return {"kind": "net1", **kwargs}

    def suite2_v1_v4_config(self, **kwargs):
        self.calls.append(("suite2_v1_v4_config", kwargs))
        return {"kind": "v1_v4", **kwargs}

    def construct(self, cfg):
        self.calls.append(("construct", cfg))
        return _FakeModel()

    def simulate(self, model, **kwargs):
        self.calls.append(("simulate", kwargs))
        return _FakeSignals()


def test_playground_request_manifest_has_truth_gates():
    spec = JaxfnePlaygroundSpec(seed=3, duration_ms=10.0, dt_ms=0.1)
    manifest = build_request_manifest(spec)

    assert manifest["canonical_import"] == "import jaxfne as jtfne"
    assert manifest["status_gates"]["truth_mode"] == "truth_safe_unverified"
    assert manifest["status_gates"]["claim_level"] == "computational_scaffold"
    assert manifest["status_gates"]["field_status"] == "laminar_proxy_no_pde"
    assert manifest["status_gates"]["physical_amplitude_claim_allowed"] is False
    assert manifest["engine_api"]["config_builder"] == "suite2_four_celltype_config"


def test_playgrounds_import_without_jaxfne_runtime_dependency():
    names = available_playgrounds()
    assert "suite2_four_celltype" in names
    report = jaxfne_backend_report(jaxfne_module=None)
    assert "available" in report
    assert report["physical_amplitude_claim_allowed"] is False


def test_build_config_uses_public_jaxfne_api_only():
    fake = _FakeJaxfne()
    spec = JaxfnePlaygroundSpec(name="suite2_single_neuron", cell_type="PV", seed=5)
    cfg = build_jaxfne_config(spec, jaxfne_module=fake)

    assert cfg["kind"] == "single"
    assert cfg["cell_type"] == "PV"
    assert fake.calls[0][0] == "suite2_single_neuron_config"


def test_run_playground_smoke_with_fake_jaxfne_module():
    fake = _FakeJaxfne()
    receipt = run_playground_smoke(JaxfnePlaygroundSpec(), jaxfne_module=fake)

    assert receipt["decision"] == "ACCEPT_SMOKE"
    assert receipt["validation"]["ok"] is True
    assert receipt["validation"]["vm_shape"] == [5, 4]
    assert receipt["validation"]["n_E_selected"] == 2
    assert [name for name, _ in fake.calls] == [
        "suite2_four_celltype_config",
        "construct",
        "simulate",
    ]


def test_validate_signal_contract_rejects_shape_mismatch():
    bad = SimpleNamespace(V_m=np.zeros((3, 2)), spk=np.zeros((3, 3)))
    report = validate_signal_contract(bad)

    assert report["ok"] is False
    assert any("shape mismatch" in msg for msg in report["errors"])


def test_invalid_spec_rejects_physical_amplitude_claim():
    spec = JaxfnePlaygroundSpec(physical_amplitude_claim_allowed=True)
    with pytest.raises(ValueError, match="physical_amplitude_claim_allowed"):
        spec.validate()
