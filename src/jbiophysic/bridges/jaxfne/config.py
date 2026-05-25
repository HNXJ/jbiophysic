"""Configuration dataclasses for jaxfne bridge.

truth_mode: truth_safe_unverified
claim_level: computational_scaffold
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Dict, Any, Literal

# Canonical constants
TRUTH_MODE = "truth_safe_unverified"
CLAIM_LEVEL = "computational_scaffold"
BRIDGE_VERSION = "bridges.jaxfne.v0.1"


@dataclass
class BridgeConfig:
    """Base bridge configuration."""

    truth_mode: str = TRUTH_MODE
    claim_level: str = CLAIM_LEVEL
    jbiophysic_bridge_version: str = BRIDGE_VERSION
    physical_amplitude_claim_allowed: bool = False
    allow_nan_in_manifest: bool = False
    seed: int = 0

    def validate(self) -> tuple[bool, list[str]]:
        """Validate configuration constants."""
        errors = []
        if self.truth_mode != TRUTH_MODE:
            errors.append(f"truth_mode must be '{TRUTH_MODE}', got {self.truth_mode}")
        if self.claim_level != CLAIM_LEVEL:
            errors.append(f"claim_level must be '{CLAIM_LEVEL}', got {self.claim_level}")
        if self.physical_amplitude_claim_allowed:
            errors.append("physical_amplitude_claim_allowed must be False in Stage 2")
        return len(errors) == 0, errors


@dataclass
class SingleNeuronConfig(BridgeConfig):
    """Single neuron run configuration."""

    cell_type: Literal["izhikevich", "hodgkin_huxley"] = "izhikevich"
    params: Dict[str, Any] = field(default_factory=dict)
    stimulus_pattern: Dict[str, Any] = field(default_factory=dict)
    duration_ms: float = 1000.0
    dt_ms: float = 0.1

    def validate(self) -> tuple[bool, list[str]]:
        """Validate single-neuron config."""
        is_valid, errors = super().validate()

        if self.cell_type not in ("izhikevich", "hodgkin_huxley"):
            errors.append(f"cell_type must be izhikevich or hodgkin_huxley, got {self.cell_type}")

        if self.duration_ms <= 0:
            errors.append(f"duration_ms must be > 0, got {self.duration_ms}")
        if self.dt_ms <= 0:
            errors.append(f"dt_ms must be > 0, got {self.dt_ms}")

        try:
            n_steps = int(round(self.duration_ms / self.dt_ms))
            if abs(n_steps * self.dt_ms - self.duration_ms) > 1e-9 * max(1, self.duration_ms):
                errors.append(
                    f"dt_ms={self.dt_ms} does not divide duration_ms={self.duration_ms} evenly; "
                    f"n_steps={n_steps}, residual={abs(n_steps * self.dt_ms - self.duration_ms)}"
                )
        except Exception as e:
            errors.append(f"Error computing n_steps: {e}")

        return len(errors) == 0, errors


@dataclass
class EINetworkConfig(BridgeConfig):
    """E/I network run configuration."""

    n_exc: int = 100
    n_inh: int = 25
    connectivity_config: Dict[str, Any] = field(default_factory=dict)
    stimulus_config: Dict[str, Any] = field(default_factory=dict)
    duration_ms: float = 2000.0
    dt_ms: float = 0.1

    def validate(self) -> tuple[bool, list[str]]:
        """Validate E/I network config."""
        is_valid, errors = super().validate()

        if self.n_exc < 0:
            errors.append(f"n_exc must be >= 0, got {self.n_exc}")
        if self.n_inh < 0:
            errors.append(f"n_inh must be >= 0, got {self.n_inh}")
        if self.n_exc + self.n_inh == 0:
            errors.append(f"total neurons (n_exc + n_inh) must be > 0")

        if self.duration_ms <= 0:
            errors.append(f"duration_ms must be > 0, got {self.duration_ms}")
        if self.dt_ms <= 0:
            errors.append(f"dt_ms must be > 0, got {self.dt_ms}")

        try:
            n_steps = int(round(self.duration_ms / self.dt_ms))
            if abs(n_steps * self.dt_ms - self.duration_ms) > 1e-9 * max(1, self.duration_ms):
                errors.append(
                    f"dt_ms={self.dt_ms} does not divide duration_ms={self.duration_ms} evenly; "
                    f"n_steps={n_steps}, residual={abs(n_steps * self.dt_ms - self.duration_ms)}"
                )
        except Exception as e:
            errors.append(f"Error computing n_steps: {e}")

        return len(errors) == 0, errors


@dataclass
class LaminarProxyConfig(BridgeConfig):
    """Laminar proxy run configuration."""

    source_scale: Literal["toy", "proxy", "calibrated", "physical"] = "proxy"
    laminar_config: Dict[str, Any] = field(default_factory=dict)
    stimulus_pattern: Dict[str, Any] = field(default_factory=dict)
    duration_ms: float = 5000.0
    dt_ms: float = 0.1

    def validate(self) -> tuple[bool, list[str]]:
        """Validate laminar proxy config."""
        is_valid, errors = super().validate()

        if self.source_scale not in ("toy", "proxy", "calibrated", "physical"):
            errors.append(
                f"source_scale must be toy/proxy/calibrated/physical, got {self.source_scale}"
            )

        if self.duration_ms <= 0:
            errors.append(f"duration_ms must be > 0, got {self.duration_ms}")
        if self.dt_ms <= 0:
            errors.append(f"dt_ms must be > 0, got {self.dt_ms}")

        try:
            n_steps = int(round(self.duration_ms / self.dt_ms))
            if abs(n_steps * self.dt_ms - self.duration_ms) > 1e-9 * max(1, self.duration_ms):
                errors.append(
                    f"dt_ms={self.dt_ms} does not divide duration_ms={self.duration_ms} evenly; "
                    f"n_steps={n_steps}, residual={abs(n_steps * self.dt_ms - self.duration_ms)}"
                )
        except Exception as e:
            errors.append(f"Error computing n_steps: {e}")

        return len(errors) == 0, errors
