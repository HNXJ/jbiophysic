"""Hodgkin-Huxley diagnostics: state validation, spike detection, stability reporting.

v0.3.3: Comprehensive checks for numerical stability, gate bounds, and biophysical validity.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np


class HHStateCheckResult(NamedTuple):
    """Result of HH state validation."""

    all_finite: bool
    V_finite: bool
    m_finite: bool
    h_finite: bool
    n_finite: bool
    gates_in_bounds: bool
    m_min: float
    m_max: float
    h_min: float
    h_max: float
    n_min: float
    n_max: float
    tolerance: float


def hh_state_check(
    V_trace: np.ndarray,
    m_trace: np.ndarray,
    h_trace: np.ndarray,
    n_trace: np.ndarray,
    tolerance: float = 1e-6,
) -> HHStateCheckResult:
    """Validate Hodgkin-Huxley state traces: finiteness and gate bounds.

    Parameters
    ----------
    V_trace : np.ndarray
        Voltage trace (mV).
    m_trace, h_trace, n_trace : np.ndarray
        Gating variable traces.
    tolerance : float, optional
        Slack for gate bounds check [0 - tolerance, 1 + tolerance].
        Default 1e-6.

    Returns
    -------
    result : HHStateCheckResult
        Finiteness checks, gate bounds, min/max values.
    """
    V_finite = bool(np.all(np.isfinite(V_trace)))
    m_finite = bool(np.all(np.isfinite(m_trace)))
    h_finite = bool(np.all(np.isfinite(h_trace)))
    n_finite = bool(np.all(np.isfinite(n_trace)))
    all_finite = V_finite and m_finite and h_finite and n_finite

    m_min, m_max = float(np.min(m_trace)), float(np.max(m_trace))
    h_min, h_max = float(np.min(h_trace)), float(np.max(h_trace))
    n_min, n_max = float(np.min(n_trace)), float(np.max(n_trace))

    gates_in_bounds = (
        (m_min >= -tolerance and m_max <= 1.0 + tolerance)
        and (h_min >= -tolerance and h_max <= 1.0 + tolerance)
        and (n_min >= -tolerance and n_max <= 1.0 + tolerance)
    )

    return HHStateCheckResult(
        all_finite=all_finite,
        V_finite=V_finite,
        m_finite=m_finite,
        h_finite=h_finite,
        n_finite=n_finite,
        gates_in_bounds=gates_in_bounds,
        m_min=m_min,
        m_max=m_max,
        h_min=h_min,
        h_max=h_max,
        n_min=n_min,
        n_max=n_max,
        tolerance=tolerance,
    )


class HHSpikeDetectionResult(NamedTuple):
    """Result of spike detection."""

    n_spikes: int
    spike_indices: list
    peak_voltage_mV: float
    min_voltage_mV: float
    after_hyperpolarization_mV: float
    threshold_mV: float


def hh_spike_detection(
    V_trace: np.ndarray,
    threshold_mV: float = 0.0,
) -> HHSpikeDetectionResult:
    """Detect spikes in voltage trace via threshold crossings.

    Parameters
    ----------
    V_trace : np.ndarray
        Voltage trace (mV).
    threshold_mV : float, optional
        Spike threshold (mV). Default 0.0.

    Returns
    -------
    result : HHSpikeDetectionResult
        Spike count, indices, peak/min voltages, after-hyperpolarization.
    """
    # Detect upward threshold crossings
    above_threshold = V_trace > threshold_mV
    crossings = np.diff(above_threshold.astype(int))
    spike_indices = np.where(crossings == 1)[0].tolist()
    n_spikes = len(spike_indices)

    peak_voltage = float(np.max(V_trace))
    min_voltage = float(np.min(V_trace))

    # After-hyperpolarization: minimum voltage after last spike
    if n_spikes > 0:
        last_spike_idx = spike_indices[-1]
        after_spike_segment = V_trace[last_spike_idx:]
        after_hyperpolarization = float(np.min(after_spike_segment))
    else:
        after_hyperpolarization = min_voltage

    return HHSpikeDetectionResult(
        n_spikes=n_spikes,
        spike_indices=spike_indices,
        peak_voltage_mV=peak_voltage,
        min_voltage_mV=min_voltage,
        after_hyperpolarization_mV=after_hyperpolarization,
        threshold_mV=threshold_mV,
    )


class HHStabilityReport(NamedTuple):
    """Comprehensive stability and validity report."""

    is_stable: bool
    all_finite: bool
    gates_valid: bool
    spike_count: int
    peak_voltage_mV: float
    after_hyperpolarization_mV: float
    stiffness_warning: bool
    stiffness_reason: str
    state_check: HHStateCheckResult
    spike_detection: HHSpikeDetectionResult


def hh_stability_report(
    V_trace: np.ndarray,
    m_trace: np.ndarray,
    h_trace: np.ndarray,
    n_trace: np.ndarray,
    I_ion_trace: np.ndarray,
    I_rhs_trace: np.ndarray,
    dt_ms: float,
    g_Na_bar: float,
    g_L: float,
    stiffness_dt_threshold_ms: float = 0.1,
    spike_threshold_mV: float = 0.0,
    gate_tolerance: float = 1e-6,
) -> HHStabilityReport:
    """Generate comprehensive HH stability and validity report.

    Parameters
    ----------
    V_trace, m_trace, h_trace, n_trace : np.ndarray
        State traces.
    I_ion_trace, I_rhs_trace : np.ndarray
        Ionic and RHS current traces.
    dt_ms : float
        Integration timestep (ms).
    g_Na_bar : float
        Max Na conductance (nS).
    g_L : float
        Leak conductance (nS).
    stiffness_dt_threshold_ms : float, optional
        Timestep threshold for stiffness warning. Default 0.1 ms.
    spike_threshold_mV : float, optional
        Spike detection threshold (mV). Default 0.0.
    gate_tolerance : float, optional
        Gate bounds tolerance. Default 1e-6.

    Returns
    -------
    report : HHStabilityReport
        Comprehensive validity report combining state and spike checks.
    """
    state_check = hh_state_check(V_trace, m_trace, h_trace, n_trace, tolerance=gate_tolerance)
    spike_detect = hh_spike_detection(V_trace, threshold_mV=spike_threshold_mV)

    # Stiffness heuristic: if dt is large relative to membrane time constant
    # and Na conductance is >> leak, flag potential instability
    stiffness_warning = False
    stiffness_reason = ""
    if dt_ms > stiffness_dt_threshold_ms and g_Na_bar > g_L:
        stiffness_warning = True
        stiffness_reason = (
            f"dt={dt_ms} ms > {stiffness_dt_threshold_ms} ms and "
            f"g_Na_bar={g_Na_bar} >> g_L={g_L}; potential instability"
        )

    is_stable = state_check.all_finite and state_check.gates_in_bounds and not stiffness_warning

    return HHStabilityReport(
        is_stable=is_stable,
        all_finite=state_check.all_finite,
        gates_valid=state_check.gates_in_bounds,
        spike_count=spike_detect.n_spikes,
        peak_voltage_mV=spike_detect.peak_voltage_mV,
        after_hyperpolarization_mV=spike_detect.after_hyperpolarization_mV,
        stiffness_warning=stiffness_warning,
        stiffness_reason=stiffness_reason,
        state_check=state_check,
        spike_detection=spike_detect,
    )
