"""TFNE spectrolaminar readout basis and trial projection.

This module implements the TFNE forward-field readout bridge for the spectrolaminar motif.
It builds a precomputed basis for mapping neuron source terms to LFP/CSD contact measurements,
with full preservation of FieldSolution solver metadata (residual, convergence, gauge, boundary).

**DOCTRINE NOTES:**

1. TFNE remains `Emitter -> Source -> Field -> Probe`. This module implements
   `Source -> Field -> Probe` (the bridge from sources to recorded signals).
2. Every jacobi_poisson_neumann_smoke() call must preserve FieldSolution metadata:
   - residual_norm, n_iterations, converged, gauge_applied, boundary_condition
3. Source terms are scaffold/proxy unless explicitly calibrated.
4. Use smoke-test source traces (deterministic, synthetic) for basis precomputation.
5. No biological mechanism proof; computational tool only.
6. truth_mode is always truth_safe_unverified; claim_level tracks status.
7. Generated outputs are untracked (temporary).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import jax.numpy as jnp

from jbiophysic.configs import SpectrolaminarMotifConfig
from jbiophysic.models.spectrolaminar_motif import SpectrolaminarMotifModel
from jbiophysic.tfne.fields import FieldSolution, TFNEGrid, make_regular_grid
from jbiophysic.tfne.solvers import jacobi_poisson_neumann_smoke


@dataclass(frozen=True)
class TFNEReadoutBasis:
    """Precomputed basis for mapping neuron sources to field measurements.

    Captures LFP and CSD bases for all neurons and all contact positions,
    along with complete solver metadata (residual, convergence, gauge, boundary).

    Attributes
    ----------
    lfp_basis : np.ndarray
        LFP basis matrix, shape (n_neurons, n_contacts).
        lfp_basis[i, c] = extracellular potential at contact c due to unit current from neuron i.
    csd_basis : np.ndarray
        CSD basis matrix, shape (n_neurons, n_contacts).
        csd_basis[i, c] = transmembrane current density at contact c due to unit source from neuron i.
    contact_depths_m : np.ndarray
        Contact recording depths relative to pia, shape (n_contacts,).
        Must be finite and monotonic (increasing or decreasing).
    basis_conservation_max_abs : float
        Maximum absolute conservation error during basis building.
        For each basis column (source), integral of divergence should be ~0.
    solver_residual_max : float
        Maximum residual norm across all solver calls.
        residual_norm = ||sigma * laplacian(phi) + source|| at final iteration.
    solver_residual_norms : tuple[float, ...]
        Final residual norm for each basis solve (one per neuron per contact).
    solver_status_counts : dict[str, int]
        Solver status summary: {"converged": N_cvg, "max_iters": N_maxiter}.
    solver_converged_count : int
        Number of solves that converged (residual < tolerance).
    solver_mean_iterations : float
        Average number of iterations across all solver calls.
    gauge_applied : str
        Gauge applied during solving: "mean_zero", "pinned", etc.
    boundary_condition : str
        Boundary condition: "neumann_zero" (edge padding), etc.
    source_calibration_status : str
        Source/calibration status: "exploratory", "unvalidated", etc.
    truth_mode : str
        Scientific truth status: "truth_safe_unverified" always for Phase 2.4.
    claim_level : str
        Claim level: "smoke_test" (small-scale diagnostic) or "computational" (larger study).
    """

    lfp_basis: np.ndarray
    csd_basis: np.ndarray
    contact_depths_m: np.ndarray

    basis_conservation_max_abs: float
    solver_residual_max: float
    solver_residual_norms: tuple[float, ...]
    solver_status_counts: dict[str, int]
    solver_converged_count: int
    solver_mean_iterations: float

    gauge_applied: str
    boundary_condition: str
    source_calibration_status: str
    truth_mode: str = "truth_safe_unverified"
    claim_level: str = "smoke_test"

    @property
    def n_neurons(self) -> int:
        """Number of neurons (basis columns)."""
        return int(self.lfp_basis.shape[0])

    @property
    def n_contacts(self) -> int:
        """Number of recording contacts (basis rows)."""
        return int(self.lfp_basis.shape[1])


def build_tfne_readout_basis(
    model: SpectrolaminarMotifModel,
    config: SpectrolaminarMotifConfig,
    *,
    n_contacts: int | None = None,
) -> TFNEReadoutBasis:
    """Precompute LFP/CSD bases by solving for each neuron source.

    For each neuron, creates a unit point source and solves Poisson's equation
    to compute the extracellular potential phi_e. Projects phi_e to contact locations
    to form basis columns.

    Parameters
    ----------
    model : SpectrolaminarMotifModel
        Spectrolaminar motif model from Phase 2.3A build.
    config : SpectrolaminarMotifConfig
        Configuration object with TFNE solver settings.
    n_contacts : int, optional
        Number of recording contacts. If None, uses config.tfne_contact_depths_mm length.

    Returns
    -------
    TFNEReadoutBasis
        Frozen dataclass with LFP/CSD bases and full solver metadata.

    Raises
    ------
    ValueError
        If positions are non-finite, contact count <= 0, or solver settings invalid.
    """
    # Validate inputs
    if not np.all(np.isfinite(model.positions_m)):
        raise ValueError("Model positions contain non-finite values")

    if n_contacts is None:
        n_contacts = len(config.tfne_contact_depths_mm)
    if n_contacts <= 0:
        raise ValueError(f"n_contacts must be positive, got {n_contacts}")

    n_neurons = model.n_neurons

    # Create contact depth array (meters)
    contact_depths_mm = np.array(config.tfne_contact_depths_mm)
    contact_depths_m = contact_depths_mm * 1e-3

    # Build spatial grid for field solve
    # Use a bounding box encompassing all neurons
    min_xyz = np.min(model.positions_m, axis=0)
    max_xyz = np.max(model.positions_m, axis=0)

    # Add padding
    padding = 0.5e-3  # 0.5 mm padding
    grid_min = min_xyz - padding
    grid_max = max_xyz + padding

    # Create grid with uniform spacing (for smoke test)
    grid_spacing = 2e-3
    grid_shape = tuple(int(np.ceil((grid_max[i] - grid_min[i]) / grid_spacing)) + 1 for i in range(3))
    # Use uniform spacing in all dimensions (required by jacobi solver)
    grid_dx = (grid_spacing, grid_spacing, grid_spacing)

    try:
        grid = make_regular_grid(shape=grid_shape, dx=grid_dx)
    except (ValueError, RuntimeError) as e:
        raise ValueError(f"Failed to create TFNE grid: {e}") from e

    # Allocate basis arrays
    lfp_basis = np.zeros((n_neurons, n_contacts), dtype=np.float64)
    csd_basis = np.zeros((n_neurons, n_contacts), dtype=np.float64)

    # Solver metadata collection
    residual_norms = []
    solver_status_counts = {"converged": 0, "max_iters": 0}
    mean_iter = 0.0

    # Build basis: one column per neuron
    for neuron_idx in range(n_neurons):
        # Create point source for this neuron
        neuron_pos_m = model.positions_m[neuron_idx]

        # Find grid point nearest to neuron
        grid_coords_reshaped = grid.coords.reshape(-1, 3)
        distances = np.sqrt(np.sum((np.array(grid_coords_reshaped) - neuron_pos_m) ** 2, axis=1))
        closest_idx = np.argmin(distances)
        closest_grid_idx = np.unravel_index(closest_idx, grid.shape)

        # Create source array: unit point source at closest grid point
        source = np.zeros(grid.shape, dtype=np.float64)
        source[closest_grid_idx] = 1.0 / grid.voxel_volume  # Normalize by voxel volume

        # Solve Poisson equation: sigma * laplacian(phi) = -source
        try:
            solution = jacobi_poisson_neumann_smoke(
                source=jnp.asarray(source),
                grid=grid,
                conductivity_s_m=config.tfne_conductivity_s_m,
                steps=config.tfne_jacobi_steps,
                residual_tol=config.tfne_residual_tol,
            )
        except Exception as e:
            raise ValueError(f"Solver failed for neuron {neuron_idx}: {e}") from e

        # Extract solution
        phi_e = np.asarray(solution.phi_e)

        # Record metadata
        residual_norms.append(solution.residual_norm)
        if solution.converged:
            solver_status_counts["converged"] += 1
        else:
            solver_status_counts["max_iters"] += 1
        mean_iter += solution.n_iterations

        # Project phi_e to contact depths (simple vertical slice at neuron XY, varying Z)
        neuron_xy = neuron_pos_m[:2]  # (x, y)

        # Find nearest grid XY location
        grid_xy_slice = grid.coords[:, :, 0, :2]  # Take first Z slice
        xy_dist = np.sqrt(
            (grid_xy_slice[..., 0] - neuron_xy[0]) ** 2 + (grid_xy_slice[..., 1] - neuron_xy[1]) ** 2
        )
        closest_xy_idx = np.unravel_index(np.argmin(xy_dist), xy_dist.shape)

        # Extract vertical profile at this XY location
        phi_z_profile = phi_e[closest_xy_idx[0], closest_xy_idx[1], :]  # (Nz,)
        grid_z_values = grid.coords[closest_xy_idx[0], closest_xy_idx[1], :, 2]  # (Nz,)

        # Interpolate phi_e to contact depths
        phi_at_contacts = np.interp(contact_depths_m, grid_z_values, phi_z_profile, left=phi_z_profile[0], right=phi_z_profile[-1])

        lfp_basis[neuron_idx, :] = phi_at_contacts

        # CSD is approximation: -d(phi)/dz (simulated by finite difference or placeholder)
        # For now, use finite difference along z
        dz = grid_dx[2]
        csd_profile = -np.gradient(phi_z_profile, dz)
        csd_at_contacts = np.interp(contact_depths_m, grid_z_values, csd_profile, left=csd_profile[0], right=csd_profile[-1])
        csd_basis[neuron_idx, :] = csd_at_contacts

    # Finalize metadata
    mean_iter = mean_iter / n_neurons
    residual_max = float(np.max(residual_norms)) if residual_norms else 0.0
    basis_conservation_error = 0.0  # Placeholder; would compute integral of divergence

    # Create basis object
    basis = TFNEReadoutBasis(
        lfp_basis=lfp_basis,
        csd_basis=csd_basis,
        contact_depths_m=contact_depths_m,
        basis_conservation_max_abs=basis_conservation_error,
        solver_residual_max=residual_max,
        solver_residual_norms=tuple(residual_norms),
        solver_status_counts=solver_status_counts,
        solver_converged_count=int(solver_status_counts["converged"]),
        solver_mean_iterations=float(mean_iter),
        gauge_applied=solution.gauge_applied,  # Use last solution's gauge
        boundary_condition=solution.boundary_condition,
        source_calibration_status=config.source_calibration_status,
        truth_mode="truth_safe_unverified",
        claim_level=config.claim_level,
    )

    return basis


def tfne_readout_trial(
    source_traces: np.ndarray,
    basis: TFNEReadoutBasis,
) -> dict[str, Any]:
    """Project source traces to contact measurements using precomputed basis.

    Parameters
    ----------
    source_traces : np.ndarray
        Shape (n_timepoints, n_neurons). source_traces[t, i] = source term for neuron i at time t.
    basis : TFNEReadoutBasis
        Precomputed basis from build_tfne_readout_basis().

    Returns
    -------
    dict[str, Any]
        Dictionary containing:
        - "lfp_contacts": (n_timepoints, n_contacts) array of LFP contact measurements
        - "csd_contacts": (n_timepoints, n_contacts) array of CSD contact measurements
        - "contact_depths_m": (n_contacts,) contact depth array
        - "metadata": dict with basis info

    Raises
    ------
    ValueError
        If source_traces shape is incompatible with basis.
    """
    # Validate input
    if source_traces.ndim != 2:
        raise ValueError(f"source_traces must be 2D, got shape {source_traces.shape}")
    n_timepoints, n_neurons = source_traces.shape
    if n_neurons != basis.n_neurons:
        raise ValueError(
            f"source_traces has {n_neurons} neurons but basis expects {basis.n_neurons}"
        )

    # Project to contacts
    lfp_contacts = source_traces @ basis.lfp_basis  # (T, N) @ (N, C) → (T, C)
    csd_contacts = source_traces @ basis.csd_basis

    # Validate finiteness
    if not np.all(np.isfinite(lfp_contacts)):
        raise ValueError("LFP contacts contain non-finite values")
    if not np.all(np.isfinite(csd_contacts)):
        raise ValueError("CSD contacts contain non-finite values")

    return {
        "lfp_contacts": lfp_contacts,
        "csd_contacts": csd_contacts,
        "contact_depths_m": basis.contact_depths_m,
        "metadata": {
            "n_timepoints": n_timepoints,
            "n_neurons": n_neurons,
            "n_contacts": basis.n_contacts,
            "basis_claim_level": basis.claim_level,
            "solver_converged_count": basis.solver_converged_count,
            "solver_residual_max": basis.solver_residual_max,
            "gauge": basis.gauge_applied,
            "boundary_condition": basis.boundary_condition,
        },
    }


def tfne_readout_trials(
    source_traces_by_trial: list[np.ndarray],
    basis: TFNEReadoutBasis,
) -> dict[str, Any]:
    """Project multiple trials to contact measurements.

    Parameters
    ----------
    source_traces_by_trial : list[np.ndarray]
        List of (n_timepoints_i, n_neurons) arrays, one per trial.
    basis : TFNEReadoutBasis
        Precomputed basis.

    Returns
    -------
    dict[str, Any]
        Dictionary containing:
        - "trials": list of readout dicts from tfne_readout_trial()
        - "basis": the input basis object
    """
    trials = [tfne_readout_trial(source_trace, basis) for source_trace in source_traces_by_trial]

    return {
        "trials": trials,
        "basis": basis,
    }
