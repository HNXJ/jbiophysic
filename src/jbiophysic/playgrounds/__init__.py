"""Playground adapters for external neural-circuit engines.

The playground namespace is intentionally thin. It stores jbiophysic-side
contracts, manifests, and smoke validation around optional engines without
copying their simulation kernels.
"""

from __future__ import annotations

from .jaxfne_networks import (
    JaxfnePlaygroundSpec,
    available_playgrounds,
    build_jaxfne_config,
    build_request_manifest,
    construct_jaxfne_model,
    jaxfne_backend_report,
    require_jaxfne,
    run_playground_smoke,
    simulate_jaxfne_model,
    validate_signal_contract,
    write_playground_receipt,
)

__all__ = [
    "JaxfnePlaygroundSpec",
    "available_playgrounds",
    "build_jaxfne_config",
    "build_request_manifest",
    "construct_jaxfne_model",
    "jaxfne_backend_report",
    "require_jaxfne",
    "run_playground_smoke",
    "simulate_jaxfne_model",
    "validate_signal_contract",
    "write_playground_receipt",
]
