"""Configuration for TFNE-Izhikevich spectrolaminar motif scaffold.

This module defines the SpectrolaminarMotifConfig frozen dataclass,
which captures all parameters for the three-area V1→V4→PFC laminar
circuit model with TFNE forward-field readout.

**IMPORTANT DOCTRINE NOTES:**

1. This scaffold is **exploratory** and **not a biological validation**.
2. Izhikevich parameters are adapted for tutorial purposes.
3. TFNE readout is a forward-field modeling tool, not a verified head model.
4. All scientific claims require truth_safe_unverified status.
5. Generated simulation outputs are untracked (temporary only).
6. All neurons are point sources, not morphologies.
7. Spectrolaminar target is a readout objective, not optimized generator state.
8. EEG/MEG are toy projections without validated head/volume conductor models.

See jbiophysic/docs for full doctrine.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Literal

import yaml
import numpy as np


@dataclass(frozen=True)
class SpectrolaminarMotifConfig:
    """Configuration for the TFNE-Izhikevich spectrolaminar motif scaffold.

    This frozen dataclass captures all parameters needed to construct and simulate
    a three-area (V1 → V4 → PFC) laminar cortical network with TFNE
    forward-field readout and spectrolaminar analysis.

    Parameters
    ----------
    mode : Literal["smoke", "full"]
        Execution mode. "smoke" is small/fast for testing; "full" is larger
        intended parameters but may be slow in CI.
    seed : int
        Random seed for reproducibility.
    truth_mode : Literal["truth_safe_unverified"]
        Scientific truth status. Always "truth_safe_unverified" for this
        exploratory scaffold.
    dt_ms : float
        Simulation timestep in milliseconds. Must be > 0.
    t_start_ms : float
        Simulation start time in milliseconds.
    t_stop_ms : float
        Simulation stop time in milliseconds. Must be > t_start_ms.
    stimulus_onset_ms : float
        Stimulus event onset time relative to trial start.
    stimulus_duration_ms : float
        Stimulus event duration in milliseconds.
    baseline_duration_ms : float
        Pre-stimulus baseline window duration.
    post_duration_ms : float
        Post-stimulus window duration.
    neurons_per_area_per_class : dict[str, int]
        Total neuron counts by area: {"V1": n, "V4": m, "PFC": k}.
    cell_counts_by_class : dict[str, float]
        Fractional cell counts as {"E": frac_e, "PV": frac_pv, "SST": frac_sst, "VIP": frac_vip}.
        Must sum to 1.0.
    laminar_fractions : dict[str, float]
        Fractional laminar distribution as {"superficial": f0, "middle": f1, "deep": f2}.
        Must sum to 1.0.
    laminar_depths_mm : dict[str, tuple[float, float]]
        Depth bounds per layer as {"superficial": (z0, z1), "middle": (z2, z3), "deep": (z4, z5)}.
    neuron_xy_radius_mm : float
        Horizontal (XY) spatial extent per area, in millimeters.
    neuron_min_distance_um : float
        Minimum inter-neuron distance constraint, in micrometers.
    izhikevich_param_preset : Literal["hodgkin_huxley_like", "regular_spiking", "fast_spiking"]
        Izhikevich parameter preset family.
    connectivity_rule : Literal["all_to_all", "sparse_laminar"]
        Connectivity rule: "all_to_all" for within-layer, sparse for multi-area.
    connection_probability : float
        Probability of connection in sparse rules. Default 0.1.
    tfne_conductivity_s_m : float
        Conductivity in Siemens/meter for TFNE field solve.
    tfne_jacobi_steps : int
        Maximum Jacobi iteration steps for TFNE field solver.
    tfne_residual_tol : float
        Residual tolerance for early exit in TFNE field solver.
    tfne_boundary_condition : Literal["neumann_zero"]
        Boundary condition for TFNE field solver.
    tfne_gauge : Literal["mean_zero"]
        Gauge condition for TFNE field solver.
    tfne_contact_depths_mm : list[float]
        Recording contact depths relative to pia, in millimeters.
    plasticity_coefficient : float
        Global synaptic plasticity scale (dimensionless gain).
    noise_level_pA : float
        Background synaptic noise level in picoamperes.
    readout_objective : Literal["spectrolaminar_alpha_gamma", "none"]
        Readout objective for analysis/optimization targets.
    claim_level : Literal["smoke_test", "computational"]
        Scientific claim level: "smoke_test" for diagnostic runs, "computational" for larger studies.
    source_calibration_status : Literal["unvalidated", "exploratory"]
        Status of source-to-field calibration.
    disclaimer : str
        Explicit disclaimer text for human readers.
    output_save_spikes : bool
        Whether to save spike raster outputs.
    output_save_rates : bool
        Whether to save firing rates.
    output_save_lfp : bool
        Whether to save LFP readout.
    output_save_csd : bool
        Whether to save CSD readout.
    output_save_field_snapshots : bool
        Whether to save field snapshots during TFNE solve.
    output_dir : str
        Output directory for generated artifacts (relative path).
    """

    # Execution mode and metadata
    mode: Literal["smoke", "full"] = "smoke"
    seed: int = 42
    truth_mode: Literal["truth_safe_unverified"] = "truth_safe_unverified"

    # Simulation timing (milliseconds)
    dt_ms: float = 0.1
    t_start_ms: float = 0.0
    t_stop_ms: float = 5000.0
    stimulus_onset_ms: float = 500.0
    stimulus_duration_ms: float = 200.0
    baseline_duration_ms: float = 500.0
    post_duration_ms: float = 1000.0

    # Population sizes
    neurons_per_area_per_class: dict[str, int] = field(default_factory=lambda: {"V1": 200, "V4": 200, "PFC": 200})
    cell_counts_by_class: dict[str, float] = field(default_factory=lambda: {"E": 0.75, "PV": 0.10, "SST": 0.10, "VIP": 0.05})
    laminar_fractions: dict[str, float] = field(default_factory=lambda: {"superficial": 0.35, "middle": 0.30, "deep": 0.35})
    laminar_depths_mm: dict[str, tuple[float, float]] = field(
        default_factory=lambda: {
            "superficial": (0.0, 0.5),
            "middle": (0.5, 0.8),
            "deep": (0.8, 1.5),
        }
    )

    # Geometry (millimeters)
    neuron_xy_radius_mm: float = 0.5
    neuron_min_distance_um: float = 10.0

    # Izhikevich parameters
    izhikevich_param_preset: Literal["hodgkin_huxley_like", "regular_spiking", "fast_spiking"] = "regular_spiking"

    # Connectivity
    connectivity_rule: Literal["all_to_all", "sparse_laminar"] = "all_to_all"
    connection_probability: float = 0.1

    # TFNE field solver settings
    tfne_conductivity_s_m: float = 0.3
    tfne_jacobi_steps: int = 1000
    tfne_residual_tol: float = 1e-6
    tfne_boundary_condition: Literal["neumann_zero"] = "neumann_zero"
    tfne_gauge: Literal["mean_zero"] = "mean_zero"
    tfne_contact_depths_mm: list[float] = field(default_factory=lambda: [0.0, 0.5, 1.0, 1.5])

    # Dynamics and readout
    plasticity_coefficient: float = 0.1
    noise_level_pA: float = 10.0
    readout_objective: Literal["spectrolaminar_alpha_gamma", "none"] = "spectrolaminar_alpha_gamma"

    # Scientific/epistemological status
    claim_level: Literal["smoke_test", "computational"] = "smoke_test"
    source_calibration_status: Literal["unvalidated", "exploratory"] = "exploratory"
    disclaimer: str = (
        "EXPLORATORY SCAFFOLD ONLY. Not biological validation. "
        "Izhikevich native current, TFNE forward-field, computational. "
        "No head model. No biophysics guarantee."
    )

    # Output toggles
    output_save_spikes: bool = True
    output_save_rates: bool = True
    output_save_lfp: bool = True
    output_save_csd: bool = True
    output_save_field_snapshots: bool = False
    output_dir: str = "/tmp/spectrolaminar_output"

    def __post_init__(self) -> None:
        """Validate configuration after initialization."""
        self._validate()

    @classmethod
    def from_yaml(cls, path: str | Path, mode: Literal["smoke", "full"] = "smoke") -> SpectrolaminarMotifConfig:
        """Load configuration from YAML file and apply mode overrides.

        Parameters
        ----------
        path : str or Path
            Path to YAML configuration file.
        mode : {"smoke", "full"}
            Execution mode to apply. Overrides mode in YAML if present.

        Returns
        -------
        SpectrolaminarMotifConfig
            Loaded and validated configuration.

        Raises
        ------
        FileNotFoundError
            If YAML file does not exist.
        KeyError
            If required fields are missing.
        ValueError
            If validation fails (e.g., invalid mode, timing bounds).
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")

        with open(path, "r") as f:
            data = yaml.safe_load(f)

        if data is None:
            data = {}

        # Extract mode-specific overrides if present
        mode_overrides = {}
        if "modes" in data and mode in data["modes"]:
            mode_overrides = data["modes"][mode]

        # Merge: defaults < base config < mode overrides
        base_data = {k: v for k, v in data.items() if k not in ["modes", "metadata"]}
        merged = {**base_data, **mode_overrides, "mode": mode}

        # Apply mode to configuration before creating instance
        # __post_init__ will call _validate() automatically
        config = cls(**merged)
        return config

    def to_dict(self) -> dict[str, Any]:
        """Serialize configuration to dictionary with deterministic ordering.

        Returns
        -------
        dict[str, Any]
            Deterministic dictionary representation.
        """
        return asdict(self)

    def with_smoke_defaults(self) -> SpectrolaminarMotifConfig:
        """Return a copy with smoke-test sized populations (fast for testing).

        Returns
        -------
        SpectrolaminarMotifConfig
            New config with smaller neuron counts and looser tolerances.
        """
        from dataclasses import replace

        return replace(
            self,
            mode="smoke",
            neurons_per_area_per_class={"V1": 50, "V4": 50, "PFC": 50},
            tfne_jacobi_steps=100,
            tfne_residual_tol=1e-4,
            claim_level="smoke_test",
        )

    def _validate(self) -> None:
        """Validate all numeric bounds and consistency constraints.

        Raises
        ------
        ValueError
            If any constraint is violated.
        """
        # Timing validation
        if self.dt_ms <= 0.0:
            raise ValueError(f"dt_ms must be positive, got {self.dt_ms}")
        if self.t_stop_ms <= self.t_start_ms:
            raise ValueError(f"t_stop_ms ({self.t_stop_ms}) must be > t_start_ms ({self.t_start_ms})")
        if self.stimulus_duration_ms <= 0.0:
            raise ValueError(f"stimulus_duration_ms must be positive, got {self.stimulus_duration_ms}")

        # Population validation
        total_neurons = sum(self.neurons_per_area_per_class.values())
        if total_neurons <= 0:
            raise ValueError(f"total neurons must be > 0, got {total_neurons}")

        cell_sum = sum(self.cell_counts_by_class.values())
        if not np.isclose(cell_sum, 1.0, atol=1e-6):
            raise ValueError(f"cell_counts_by_class must sum to 1.0, got {cell_sum}")

        laminar_sum = sum(self.laminar_fractions.values())
        if not np.isclose(laminar_sum, 1.0, atol=1e-6):
            raise ValueError(f"laminar_fractions must sum to 1.0, got {laminar_sum}")

        # Geometry validation
        if self.neuron_xy_radius_mm <= 0.0:
            raise ValueError(f"neuron_xy_radius_mm must be positive, got {self.neuron_xy_radius_mm}")
        if self.neuron_min_distance_um <= 0.0:
            raise ValueError(f"neuron_min_distance_um must be positive, got {self.neuron_min_distance_um}")

        # TFNE solver validation
        if self.tfne_jacobi_steps <= 0:
            raise ValueError(f"tfne_jacobi_steps must be positive, got {self.tfne_jacobi_steps}")
        if self.tfne_residual_tol <= 0.0:
            raise ValueError(f"tfne_residual_tol must be positive, got {self.tfne_residual_tol}")

        # Plasticity and noise
        if self.plasticity_coefficient < 0.0:
            raise ValueError(f"plasticity_coefficient must be >= 0, got {self.plasticity_coefficient}")
        if self.noise_level_pA < 0.0:
            raise ValueError(f"noise_level_pA must be >= 0, got {self.noise_level_pA}")

        # Mode validation
        if self.mode not in ("smoke", "full"):
            raise ValueError(f"unknown mode: {self.mode!r}")

        # Boundary/gauge validation
        if self.tfne_boundary_condition not in ("neumann_zero",):
            raise ValueError(f"unsupported boundary_condition: {self.tfne_boundary_condition!r}")
        if self.tfne_gauge not in ("mean_zero",):
            raise ValueError(f"unsupported gauge: {self.tfne_gauge!r}")
