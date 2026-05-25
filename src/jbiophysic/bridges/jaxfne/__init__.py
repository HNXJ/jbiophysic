"""jaxfne bridge: canonical dispatch layer between jbiophysic and jaxfne engine.

Public API:
- Config classes: BridgeConfig, SingleNeuronConfig, EINetworkConfig, LaminarProxyConfig
- Converters: jbiophysic_params_to_jaxfne, jbiophysic_circuit_to_jaxfne
- Builders: build_single_neuron_run, build_ei_network_run, build_laminar_proxy_run
- Dispatch: get_jaxfne_report, run_and_report
- I/O: json_safe, write_manifest, harmonize_jaxfne_output, get_installed_jaxfne_version
- Validation: validate_report, validate_manifest_json

truth_mode: truth_safe_unverified
claim_level: computational_scaffold
physical_amplitude_claim_allowed: False (Stage 2)
"""

from __future__ import annotations

from .config import (
    BridgeConfig,
    EINetworkConfig,
    LaminarProxyConfig,
    SingleNeuronConfig,
)
from .convert import (
    jbiophysic_circuit_to_jaxfne,
    jbiophysic_params_to_jaxfne,
)
from .reports import (
    get_installed_jaxfne_version,
    harmonize_jaxfne_output,
    json_safe,
    write_manifest,
)
from .run import (
    build_ei_network_run,
    build_laminar_proxy_run,
    build_single_neuron_run,
    get_jaxfne_report,
    run_and_report,
)
from .validation import (
    validate_manifest_json,
    validate_report,
)

__all__ = [
    # Config
    "BridgeConfig",
    "SingleNeuronConfig",
    "EINetworkConfig",
    "LaminarProxyConfig",
    # Convert
    "jbiophysic_params_to_jaxfne",
    "jbiophysic_circuit_to_jaxfne",
    # Run/dispatch
    "get_jaxfne_report",
    "build_single_neuron_run",
    "build_ei_network_run",
    "build_laminar_proxy_run",
    "run_and_report",
    # Reports
    "json_safe",
    "write_manifest",
    "harmonize_jaxfne_output",
    "get_installed_jaxfne_version",
    # Validation
    "validate_report",
    "validate_manifest_json",
]
