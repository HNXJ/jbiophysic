"""Unit conversion and dimensional analysis for biophysical quantities."""

from __future__ import annotations

from .conversions import (
    conductance_per_soma_area,
    mV_to_V,
    mm_to_um,
    nA_to_pA,
    nS_to_uS,
    pA_to_nA,
    tau_membrane_ms,
    um_to_mm,
    uS_to_nS,
    V_to_mV,
)
from .dtype_comparison import (
    DtypeComparisonResult,
    compare_dtype_passive_membrane,
    dtype_comparison_report,
)
from .stability import (
    finite_value_check,
    integration_stability_report,
    magnitude_diagnostics,
    monotonic_blow_up_check,
)

__all__ = [
    # Conversions
    "mV_to_V",
    "V_to_mV",
    "pA_to_nA",
    "nA_to_pA",
    "nS_to_uS",
    "uS_to_nS",
    "um_to_mm",
    "mm_to_um",
    "tau_membrane_ms",
    "conductance_per_soma_area",
    # Dtype comparison (v0.1.3)
    "DtypeComparisonResult",
    "compare_dtype_passive_membrane",
    "dtype_comparison_report",
    # Stability diagnostics (v0.1.4)
    "finite_value_check",
    "magnitude_diagnostics",
    "monotonic_blow_up_check",
    "integration_stability_report",
]
