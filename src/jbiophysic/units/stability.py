"""Stability and numerical diagnostics for biophysical simulations.

This module provides reusable diagnostics for detecting numerical instabilities,
finite value violations, and out-of-range behavior in neural simulations.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def finite_value_check(arr: np.ndarray, name: str = "array") -> dict[str, Any]:
    """
    Check for NaN and Inf in array.

    Parameters
    ----------
    arr : np.ndarray
        Array to check
    name : str
        Name for reporting

    Returns
    -------
    report : dict
        JSON-safe report with has_nan, has_inf, n_nan, n_inf
    """
    arr = np.asarray(arr)
    has_nan = bool(np.any(np.isnan(arr)))
    has_inf = bool(np.any(np.isinf(arr)))
    n_nan = int(np.sum(np.isnan(arr)))
    n_inf = int(np.sum(np.isinf(arr)))

    return {
        "name": name,
        "has_nan": has_nan,
        "has_inf": has_inf,
        "n_nan": n_nan,
        "n_inf": n_inf,
        "is_finite": not (has_nan or has_inf),
    }


def magnitude_diagnostics(
    arr: np.ndarray,
    name: str = "array",
    expected_range: tuple[float, float] | None = None,
) -> dict[str, Any]:
    """
    Compute magnitude statistics and check bounds.

    Parameters
    ----------
    arr : np.ndarray
        Array to analyze
    name : str
        Name for reporting
    expected_range : tuple[float, float], optional
        Expected min/max range (e.g., (-90, 30) for voltage in mV)

    Returns
    -------
    report : dict
        JSON-safe report with min, max, mean, std, out_of_range count
    """
    arr = np.asarray(arr, dtype=np.float64)

    report = {
        "name": name,
        "shape": tuple(arr.shape),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "abs_max": float(np.max(np.abs(arr))),
    }

    if expected_range is not None:
        min_exp, max_exp = expected_range
        out_of_range = (arr < min_exp) | (arr > max_exp)
        n_out = int(np.sum(out_of_range))
        frac_out = float(n_out / arr.size) if arr.size > 0 else 0.0

        report.update(
            {
                "expected_range": expected_range,
                "n_out_of_range": n_out,
                "frac_out_of_range": frac_out,
                "in_range": n_out == 0,
            }
        )

    return report


def monotonic_blow_up_check(
    arr: np.ndarray,
    growth_threshold: float = 2.0,
    name: str = "array",
) -> dict[str, Any]:
    """
    Check for monotonic exponential blow-up (instability indicator).

    If the magnitude is growing monotonically and exceeds growth_threshold
    times the initial magnitude, flag as potential instability.

    Parameters
    ----------
    arr : np.ndarray
        Time series to check (1D)
    growth_threshold : float
        Ratio (max_mag / initial_mag) that triggers flag
    name : str
        Name for reporting

    Returns
    -------
    report : dict
        JSON-safe report with is_blowing_up, growth_ratio, peak_index
    """
    arr = np.asarray(arr, dtype=np.float64).flatten()

    if len(arr) < 2:
        return {
            "name": name,
            "is_blowing_up": False,
            "growth_ratio": 1.0,
            "peak_index": 0,
            "reason": "insufficient_data",
        }

    abs_arr = np.abs(arr)
    initial_mag = abs_arr[0] if abs_arr[0] > 1e-6 else 1e-6
    peak_mag = np.max(abs_arr)
    peak_index = int(np.argmax(abs_arr))

    growth_ratio = float(peak_mag / initial_mag)
    is_blowing_up = growth_ratio > growth_threshold

    return {
        "name": name,
        "is_blowing_up": is_blowing_up,
        "growth_ratio": growth_ratio,
        "growth_threshold": growth_threshold,
        "peak_index": peak_index,
        "peak_magnitude": float(peak_mag),
        "initial_magnitude": float(initial_mag),
    }


def integration_stability_report(
    voltage_mV: np.ndarray,
    current_pA: np.ndarray,
    dt_ms: float,
    name: str = "simulation",
) -> dict[str, Any]:
    """
    Comprehensive stability report for a neural simulation.

    Parameters
    ----------
    voltage_mV : np.ndarray
        Voltage trace (mV)
    current_pA : np.ndarray
        Current trace (pA)
    dt_ms : float
        Timestep (ms)
    name : str
        Simulation name

    Returns
    -------
    report : dict
        JSON-safe report with multiple diagnostics
    """
    voltage_mV = np.asarray(voltage_mV, dtype=np.float64)
    current_pA = np.asarray(current_pA, dtype=np.float64)

    report = {
        "name": name,
        "dt_ms": float(dt_ms),
        "n_steps": int(len(voltage_mV)),
        "duration_ms": float((len(voltage_mV) - 1) * dt_ms),
    }

    # Finite checks
    report["voltage_finite"] = finite_value_check(voltage_mV, "voltage_mV")
    report["current_finite"] = finite_value_check(current_pA, "current_pA")

    # Magnitude checks
    report["voltage_magnitude"] = magnitude_diagnostics(
        voltage_mV, "voltage_mV", expected_range=(-100, 50)
    )
    report["current_magnitude"] = magnitude_diagnostics(
        current_pA, "current_pA", expected_range=(-1000, 1000)
    )

    # Blow-up checks
    report["voltage_blow_up"] = monotonic_blow_up_check(
        voltage_mV, growth_threshold=5.0, name="voltage_mV"
    )
    report["current_blow_up"] = monotonic_blow_up_check(
        current_pA, growth_threshold=100.0, name="current_pA"
    )

    # Overall stability assessment
    all_finite = report["voltage_finite"]["is_finite"] and report["current_finite"]["is_finite"]
    voltage_in_range = report["voltage_magnitude"]["in_range"]
    no_blow_up = (
        not report["voltage_blow_up"]["is_blowing_up"]
        and not report["current_blow_up"]["is_blowing_up"]
    )

    report["is_stable"] = bool(all_finite and voltage_in_range and no_blow_up)
    report["truth_mode"] = "truth_safe_unverified"
    report["claim_level"] = "numerical_stability_diagnostic"

    return report
