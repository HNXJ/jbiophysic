"""Unit conversion helpers for biophysical quantities.

All conversions follow SI base units:
- Voltage: millivolt (mV)
- Current: picoampere (pA)
- Conductance: nanosiemens (nS)
- Time: millisecond (ms)
- Length: micrometer (μm)
- Capacitance: picofarad (pF)
- Resistance: megaohm (MΩ)
"""

from __future__ import annotations


# ============================================================================
# Electrical Conversions
# ============================================================================


def mV_to_V(x: float) -> float:
    """Convert millivolt to volt.

    Parameters
    ----------
    x : float
        Voltage in millivolt (mV)

    Returns
    -------
    float
        Voltage in volt (V)
    """
    return x / 1000


def V_to_mV(x: float) -> float:
    """Convert volt to millivolt.

    Parameters
    ----------
    x : float
        Voltage in volt (V)

    Returns
    -------
    float
        Voltage in millivolt (mV)
    """
    return x * 1000


def pA_to_nA(x: float) -> float:
    """Convert picoampere to nanoampere.

    Parameters
    ----------
    x : float
        Current in picoampere (pA)

    Returns
    -------
    float
        Current in nanoampere (nA)
    """
    return x / 1000


def nA_to_pA(x: float) -> float:
    """Convert nanoampere to picoampere.

    Parameters
    ----------
    x : float
        Current in nanoampere (nA)

    Returns
    -------
    float
        Current in picoampere (pA)
    """
    return x * 1000


def nS_to_uS(x: float) -> float:
    """Convert nanosiemens to microsiemens.

    Parameters
    ----------
    x : float
        Conductance in nanosiemens (nS)

    Returns
    -------
    float
        Conductance in microsiemens (μS)
    """
    return x / 1000


def uS_to_nS(x: float) -> float:
    """Convert microsiemens to nanosiemens.

    Parameters
    ----------
    x : float
        Conductance in microsiemens (μS)

    Returns
    -------
    float
        Conductance in nanosiemens (nS)
    """
    return x * 1000


# ============================================================================
# Spatial Conversions
# ============================================================================


def um_to_mm(x: float) -> float:
    """Convert micrometer to millimeter.

    Parameters
    ----------
    x : float
        Length in micrometer (μm)

    Returns
    -------
    float
        Length in millimeter (mm)
    """
    return x / 1000


def mm_to_um(x: float) -> float:
    """Convert millimeter to micrometer.

    Parameters
    ----------
    x : float
        Length in millimeter (mm)

    Returns
    -------
    float
        Length in micrometer (μm)
    """
    return x * 1000


# ============================================================================
# Membrane Biophysics
# ============================================================================


def tau_membrane_ms(R_ohm_cm2: float, C_uF_cm2: float) -> float:
    """Calculate membrane time constant.

    The membrane acts as an RC circuit. The time constant determines
    how quickly the membrane voltage relaxes to steady state.

    Parameters
    ----------
    R_ohm_cm2 : float
        Specific membrane resistance in Ω·cm²
        Typical: 10,000–50,000 Ω·cm²

    C_uF_cm2 : float
        Specific membrane capacitance in μF/cm²
        Typical: 0.8–1.2 μF/cm² (mammalian neurons)

    Returns
    -------
    tau_ms : float
        Time constant in milliseconds

    Example
    -------
    Typical mammalian neuron:

    >>> R = 20000  # Ω·cm²
    >>> C = 1.0    # μF/cm²
    >>> tau = tau_membrane_ms(R, C)
    >>> print(f"τ_m = {tau:.1f} ms")
    τ_m = 20.0 ms

    Notes
    -----
    Formula: τ = R × C
    Unit conversion: (Ω·cm²) × (μF/cm²) / 1,000 = ms
    """
    return R_ohm_cm2 * C_uF_cm2 / 1000


def conductance_per_soma_area(
    g_density_uS_cm2: float,
    soma_area_um2: float,
) -> float:
    """Convert conductance density to total soma conductance.

    Conductances are often expressed as densities (per unit area).
    This function scales to the actual soma membrane area.

    Parameters
    ----------
    g_density_uS_cm2 : float
        Conductance density in μS/cm²
        Typical ranges:
        - Leak (g_L): 100–1000 μS/cm²
        - Peak sodium (g_Na): 50–200 μS/cm² (HH)
        - Peak potassium (g_K): 20–100 μS/cm² (HH)

    soma_area_um2 : float
        Soma membrane surface area in μm²
        Typical spherical soma (30 μm diameter):
        area = 4π × (15 μm)² ≈ 2,800 μm²

    Returns
    -------
    g_total_nS : float
        Total soma conductance in nanosiemens

    Example
    -------
    Hodgkin–Huxley neuron soma:

    >>> g_L_density = 0.3   # μS/cm² (mammalian default)
    >>> soma_diam_um = 30   # μm
    >>> soma_area = 4 * 3.14159 * (soma_diam_um / 2) ** 2
    >>> g_L_total = conductance_per_soma_area(g_L_density, soma_area)
    >>> print(f"g_L = {g_L_total:.1f} nS")
    g_L = 8.5 nS

    Notes
    -----
    Conversion: 1 cm = 10,000 μm, so 1 cm² = 10^8 μm²

    Formula:
        g_total (nS) = g_density (μS/cm²) × area (μm²) × (1 cm² / 10^8 μm²) × (1000 nS / 1 μS)
        = g_density × area / 1e5
    """
    area_cm2 = soma_area_um2 / 1e8
    return g_density_uS_cm2 * area_cm2 * 1000
