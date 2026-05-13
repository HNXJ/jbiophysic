"""Lightweight builders for the TFNE-Izhikevich spectrolaminar motif scaffold.

This module provides deterministic, testable model construction without
expensive simulation or visualization. Builders return simple arrays and
dataclasses with stable shapes suitable for later integration with
TFNE readout, optimization, and analysis modules.

**DOCTRINE:**

- This is a computational scaffold, not a biological model.
- Population sizes and geometries are tunable parameters only.
- Connectivity is all-to-all (within layer) or feedforward (across areas).
- All builders are deterministic given config + seed.
- No autapses by default.
- FieldSolution metadata must be preserved when TFNE readout is added.
- Generated outputs are untracked.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from jbiophysic.configs import SpectrolaminarMotifConfig


@dataclass(frozen=True)
class SpectrolaminarMotifModel:
    """Frozen dataclass holding static population and connectivity state.

    Attributes
    ----------
    config : SpectrolaminarMotifConfig
        Original configuration used to build this model.
    neuron_table : np.ndarray
        Structured array with columns:
        - id: int (0 to n_neurons-1)
        - cell_type: int (0=E, 1=PV, 2=SST, 3=VIP)
        - area: int (0=V1, 1=V4, 2=PFC)
        - layer: int (0=superficial, 1=middle, 2=deep)
        - x_mm, y_mm, z_mm: float (3D position in millimeters)
    positions_mm : np.ndarray
        (n_neurons, 3) array of (x, y, z) positions in millimeters.
    positions_m : np.ndarray
        (n_neurons, 3) array of positions in meters.
    cell_type_index : np.ndarray
        (n_neurons,) array of cell type indices [0, 3].
    area_index : np.ndarray
        (n_neurons,) array of area indices [0, 2].
    layer_index : np.ndarray
        (n_neurons,) array of layer indices [0, 2].
    connectivity_matrix : np.ndarray
        (n_neurons, n_neurons) sparse connectivity matrix or dense adjacency.
        connectivity_matrix[i, j] = 1 if neuron i → j, else 0.
        Main diagonal is 0 (no autapses).
    n_neurons : int
        Total number of neurons.
    n_autapses : int
        Number of autapses (should be 0).
    """

    config: SpectrolaminarMotifConfig
    neuron_table: np.ndarray
    positions_mm: np.ndarray
    positions_m: np.ndarray
    cell_type_index: np.ndarray
    area_index: np.ndarray
    layer_index: np.ndarray
    connectivity_matrix: np.ndarray

    @property
    def n_neurons(self) -> int:
        """Total number of neurons in model."""
        return len(self.neuron_table)

    @property
    def n_autapses(self) -> int:
        """Number of autapses (should always be 0)."""
        return int(np.sum(np.diag(self.connectivity_matrix)))

    @property
    def cell_type_names(self) -> np.ndarray:
        """Cell type name for each neuron as object array."""
        type_map = np.array(["E", "PV", "SST", "VIP"], dtype=object)
        return type_map[self.cell_type_index]

    @property
    def area_names(self) -> np.ndarray:
        """Area name for each neuron as object array."""
        area_map = np.array(["V1", "V4", "PFC"], dtype=object)
        return area_map[self.area_index]

    @property
    def layer_names(self) -> np.ndarray:
        """Layer name for each neuron as object array."""
        layer_map = np.array(["superficial", "middle", "deep"], dtype=object)
        return layer_map[self.layer_index]


def build_spectrolaminar_population(
    config: SpectrolaminarMotifConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build neuron populations with laminar and area assignments.

    Returns deterministic population table and index arrays without
    randomization. Positions are placeholders and should be replaced
    by build_laminar_positions.

    Parameters
    ----------
    config : SpectrolaminarMotifConfig
        Configuration object.

    Returns
    -------
    neuron_table : np.ndarray
        Structured array with id, cell_type, area, layer, x_mm, y_mm, z_mm.
    cell_type_index : np.ndarray
        Cell type indices [0, 3].
    area_index : np.ndarray
        Area indices [0, 2].
    layer_index : np.ndarray
        Layer indices [0, 2].

    Raises
    ------
    ValueError
        If total neurons <= 0 or cell/layer fractions don't sum to ~1.0.
    """

    # Build population deterministically
    neuron_id = 0
    cell_type_list = []
    area_list = []
    layer_list = []
    x_list = []
    y_list = []
    z_list = []

    cell_type_names = ["E", "PV", "SST", "VIP"]
    area_names = ["V1", "V4", "PFC"]
    layer_names = ["superficial", "middle", "deep"]

    for area_idx, area_name in enumerate(area_names):
        total_for_area = config.neurons_per_area_per_class[area_name]

        # Distribute by layer (using greedy allocation to preserve total count)
        neurons_per_layer = _distribute_counts(
            total_for_area,
            list(config.laminar_fractions.values()),
            list(config.laminar_fractions.keys()),
        )

        for layer_idx, layer_name in enumerate(layer_names):
            neurons_in_layer = neurons_per_layer[layer_idx]

            # Distribute by cell type
            neurons_by_cell_type = _distribute_counts(
                neurons_in_layer,
                list(config.cell_counts_by_class.values()),
                list(config.cell_counts_by_class.keys()),
            )

            for cell_idx, cell_type_name in enumerate(cell_type_names):
                count = neurons_by_cell_type[cell_idx]
                for _ in range(count):
                    cell_type_list.append(cell_idx)
                    area_list.append(area_idx)
                    layer_list.append(layer_idx)
                    x_list.append(0.0)  # Placeholder, will be set by build_laminar_positions
                    y_list.append(0.0)
                    z_list.append(0.0)

    # Create structured array
    dtype = [
        ("id", np.int32),
        ("cell_type", np.int32),
        ("area", np.int32),
        ("layer", np.int32),
        ("x_mm", np.float64),
        ("y_mm", np.float64),
        ("z_mm", np.float64),
    ]
    neuron_table = np.zeros(len(cell_type_list), dtype=dtype)
    neuron_table["id"] = np.arange(len(cell_type_list), dtype=np.int32)
    neuron_table["cell_type"] = cell_type_list
    neuron_table["area"] = area_list
    neuron_table["layer"] = layer_list
    neuron_table["x_mm"] = x_list
    neuron_table["y_mm"] = y_list
    neuron_table["z_mm"] = z_list

    cell_type_index = np.array(cell_type_list, dtype=np.int32)
    area_index = np.array(area_list, dtype=np.int32)
    layer_index = np.array(layer_list, dtype=np.int32)

    return neuron_table, cell_type_index, area_index, layer_index


def build_laminar_positions(
    config: SpectrolaminarMotifConfig,
    neuron_table: np.ndarray,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Assign spatial positions to neurons within laminar/areal bounds.

    Parameters
    ----------
    config : SpectrolaminarMotifConfig
        Configuration object.
    neuron_table : np.ndarray
        Neuron table from build_spectrolaminar_population.
    rng : np.random.Generator, optional
        Random generator for reproducibility. If None, uses config.seed.

    Returns
    -------
    positions_mm : np.ndarray
        (n_neurons, 3) array of positions in millimeters.
    positions_m : np.ndarray
        (n_neurons, 3) array of positions in meters.

    Notes
    -----
    - X, Y are uniform random within neuron_xy_radius_mm per area.
    - Z is assigned according to laminar_depths_mm[layer].
    - No collision detection in Phase 2.3; collision avoidance deferred to Phase 2.4+.
    - Minimum distance constraint is recorded but not enforced here.
    """
    if rng is None:
        rng = np.random.default_rng(config.seed)

    n_neurons = len(neuron_table)
    positions_mm = np.zeros((n_neurons, 3), dtype=np.float64)

    layer_map = ["superficial", "middle", "deep"]
    area_map = ["V1", "V4", "PFC"]

    for i in range(n_neurons):
        area_idx = int(neuron_table["area"][i])
        layer_idx = int(neuron_table["layer"][i])

        # Assign X, Y randomly within area bounds
        r = rng.uniform(0, config.neuron_xy_radius_mm, size=2)
        positions_mm[i, 0] = r[0]
        positions_mm[i, 1] = r[1]

        # Assign Z according to layer depth
        layer_name = layer_map[layer_idx]
        z_min, z_max = config.laminar_depths_mm[layer_name]
        positions_mm[i, 2] = rng.uniform(z_min, z_max)

    # Validate positions are finite and within bounds
    if not np.all(np.isfinite(positions_mm)):
        raise ValueError("Generated positions contain non-finite values")

    positions_m = positions_mm * 1e-3  # Convert mm to m

    return positions_mm, positions_m


def build_connectivity(
    config: SpectrolaminarMotifConfig,
    neuron_table: np.ndarray,
) -> np.ndarray:
    """Build connectivity matrix (adjacency).

    For Phase 2.3, uses simple all-to-all within-layer rule:
    - Within a layer: all-to-all with diagonal zeroed (no autapses)
    - Across areas/layers: no connections (sparse rule deferred to Phase 2.4+)

    Parameters
    ----------
    config : SpectrolaminarMotifConfig
        Configuration object.
    neuron_table : np.ndarray
        Neuron table from build_spectrolaminar_population.

    Returns
    -------
    connectivity_matrix : np.ndarray
        (n_neurons, n_neurons) boolean or int adjacency matrix.
        connectivity_matrix[i, j] = 1 if i → j, else 0.
        Diagonal is always 0.
    """
    n_neurons = len(neuron_table)
    connectivity_matrix = np.zeros((n_neurons, n_neurons), dtype=np.int32)

    area_index = neuron_table["area"]
    layer_index = neuron_table["layer"]

    # Within-layer all-to-all (within same area AND layer)
    for i in range(n_neurons):
        for j in range(n_neurons):
            if i == j:
                # No autapses
                continue
            if area_index[i] == area_index[j] and layer_index[i] == layer_index[j]:
                # Same area and layer → all-to-all connection
                connectivity_matrix[i, j] = 1

    # Validate no autapses
    if np.any(np.diag(connectivity_matrix) != 0):
        raise ValueError("Connectivity matrix has autapses")

    return connectivity_matrix


def build_spectrolaminar_motif(
    config: SpectrolaminarMotifConfig,
) -> SpectrolaminarMotifModel:
    """Build complete spectrolaminar motif model.

    Orchestrates population building, spatial positioning, and connectivity
    in deterministic order. All steps are deterministic given config.

    Parameters
    ----------
    config : SpectrolaminarMotifConfig
        Configuration object.

    Returns
    -------
    SpectrolaminarMotifModel
        Frozen dataclass with complete model state.

    Raises
    ------
    ValueError
        If any builder step validation fails.
    """
    rng = np.random.default_rng(config.seed)

    # Build population table
    neuron_table, cell_type_index, area_index, layer_index = build_spectrolaminar_population(config)

    # Assign positions
    positions_mm, positions_m = build_laminar_positions(config, neuron_table, rng=rng)

    # Update neuron_table with positions
    neuron_table["x_mm"] = positions_mm[:, 0]
    neuron_table["y_mm"] = positions_mm[:, 1]
    neuron_table["z_mm"] = positions_mm[:, 2]

    # Build connectivity
    connectivity_matrix = build_connectivity(config, neuron_table)

    # Create model
    model = SpectrolaminarMotifModel(
        config=config,
        neuron_table=neuron_table,
        positions_mm=positions_mm,
        positions_m=positions_m,
        cell_type_index=cell_type_index,
        area_index=area_index,
        layer_index=layer_index,
        connectivity_matrix=connectivity_matrix,
    )

    return model


def _distribute_counts(
    total: int,
    fractions: list[float],
    names: list[str],
) -> list[int]:
    """Distribute integer total across bins using greedy allocation.

    Ensures all bins are represented and exact total is maintained.

    Parameters
    ----------
    total : int
        Total count to distribute.
    fractions : list[float]
        Fractional allocation per bin (should sum to ~1.0).
    names : list[str]
        Names for logging/debugging.

    Returns
    -------
    counts : list[int]
        Integer counts per bin (sum == total).
    """
    n_bins = len(fractions)
    fracs = np.array(fractions, dtype=float)
    fracs = fracs / fracs.sum()  # Normalize to ensure sum to 1.0

    # Greedy: allocate floor first, then distribute remainder
    counts = (fracs * total).astype(int)
    remainder = total - counts.sum()

    # Distribute remainder to bins with highest fractional parts
    fractional_parts = (fracs * total) - counts
    indices = np.argsort(-fractional_parts)[:remainder]
    counts[indices] += 1

    return counts.tolist()
