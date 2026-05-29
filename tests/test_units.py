"""Unit tests for unit conversion utilities."""


from jbiophysic.units import (
    V_to_mV,
    conductance_per_soma_area,
    mm_to_um,
    mV_to_V,
    nA_to_pA,
    nS_to_uS,
    pA_to_nA,
    tau_membrane_ms,
    um_to_mm,
    uS_to_nS,
)


class TestElectricalConversions:
    """Test electrical unit conversions."""

    def test_mV_to_V(self):
        """Test millivolt to volt conversion."""
        assert mV_to_V(0) == 0
        assert mV_to_V(1000) == 1.0
        assert mV_to_V(-70) == -0.07

    def test_V_to_mV(self):
        """Test volt to millivolt conversion."""
        assert V_to_mV(0) == 0
        assert V_to_mV(1.0) == 1000
        assert V_to_mV(-0.07) == -70

    def test_roundtrip_voltage(self):
        """Test voltage conversion roundtrip."""
        x = -65.5
        assert abs(V_to_mV(mV_to_V(x)) - x) < 1e-10

    def test_pA_to_nA(self):
        """Test picoampere to nanoampere conversion."""
        assert pA_to_nA(0) == 0
        assert pA_to_nA(1000) == 1.0
        assert pA_to_nA(-500) == -0.5

    def test_nA_to_pA(self):
        """Test nanoampere to picoampere conversion."""
        assert nA_to_pA(0) == 0
        assert nA_to_pA(1.0) == 1000
        assert nA_to_pA(-0.5) == -500

    def test_roundtrip_current(self):
        """Test current conversion roundtrip."""
        x = -123.45
        assert abs(nA_to_pA(pA_to_nA(x)) - x) < 1e-10

    def test_nS_to_uS(self):
        """Test nanosiemens to microsiemens conversion."""
        assert nS_to_uS(0) == 0
        assert nS_to_uS(1000) == 1.0
        assert nS_to_uS(250) == 0.25

    def test_uS_to_nS(self):
        """Test microsiemens to nanosiemens conversion."""
        assert uS_to_nS(0) == 0
        assert uS_to_nS(1.0) == 1000
        assert uS_to_nS(0.25) == 250

    def test_roundtrip_conductance(self):
        """Test conductance conversion roundtrip."""
        x = 42.7
        assert abs(uS_to_nS(nS_to_uS(x)) - x) < 1e-10


class TestSpatialConversions:
    """Test spatial unit conversions."""

    def test_um_to_mm(self):
        """Test micrometer to millimeter conversion."""
        assert um_to_mm(0) == 0
        assert um_to_mm(1000) == 1.0
        assert um_to_mm(500) == 0.5

    def test_mm_to_um(self):
        """Test millimeter to micrometer conversion."""
        assert mm_to_um(0) == 0
        assert mm_to_um(1.0) == 1000
        assert mm_to_um(0.5) == 500

    def test_roundtrip_length(self):
        """Test length conversion roundtrip."""
        x = 42.3
        assert abs(mm_to_um(um_to_mm(x)) - x) < 1e-10


class TestMembraneProperties:
    """Test membrane biophysics conversions."""

    def test_tau_membrane_typical(self):
        """Test time constant for typical mammalian neuron."""
        # Typical values: R = 20,000 Ω·cm², C = 1.0 μF/cm²
        R = 20000
        C = 1.0
        tau = tau_membrane_ms(R, C)
        assert abs(tau - 20.0) < 0.01

    def test_tau_membrane_zero_resistance(self):
        """Test time constant with zero resistance (short circuit)."""
        tau = tau_membrane_ms(0, 1.0)
        assert tau == 0

    def test_tau_membrane_zero_capacitance(self):
        """Test time constant with zero capacitance (ideal resistor)."""
        tau = tau_membrane_ms(20000, 0)
        assert tau == 0

    def test_tau_membrane_high_resistance(self):
        """Test time constant with high resistance (slow)."""
        R_low = 10000
        R_high = 50000
        C = 1.0

        tau_low = tau_membrane_ms(R_low, C)
        tau_high = tau_membrane_ms(R_high, C)

        assert tau_high > tau_low
        assert abs(tau_high / tau_low - 5.0) < 0.01

    def test_tau_membrane_high_capacitance(self):
        """Test time constant with high capacitance (slow)."""
        R = 20000
        C_low = 0.5
        C_high = 2.0

        tau_low = tau_membrane_ms(R, C_low)
        tau_high = tau_membrane_ms(R, C_high)

        assert tau_high > tau_low
        assert abs(tau_high / tau_low - 4.0) < 0.01

    def test_conductance_per_soma_area(self):
        """Test conductance scaling by soma area.

        Hodgkin-Huxley leak conductance: g_L = 0.3 mS/cm² (mammalian default)
        A soma with 2,800 μm² area should have:

        g_L (mS/cm²) = 0.3 mS/cm² = 300 μS/cm²
        g_total = 300 μS/cm² × (2800 μm² / 10^8 μm²/cm²) × 1000 nS/μS
                = 300 × 0.000028 × 1000 nS
                = 8.4 nS
        """
        g_density = 300  # μS/cm² (0.3 mS/cm² converted)
        soma_area = 2800  # μm² (sphere, d=30 μm)

        g_total = conductance_per_soma_area(g_density, soma_area)

        # Expected: ~8.4 nS for HH leak conductance on typical soma
        assert abs(g_total - 8.4) < 0.1

    def test_conductance_per_soma_area_scaling(self):
        """Test that conductance scales linearly with soma area."""
        g_density = 0.5  # μS/cm²
        area1 = 1000  # μm²
        area2 = 2000  # μm² (2x larger)

        g1 = conductance_per_soma_area(g_density, area1)
        g2 = conductance_per_soma_area(g_density, area2)

        assert abs(g2 / g1 - 2.0) < 0.01

    def test_conductance_per_soma_area_zero_density(self):
        """Test conductance with zero density."""
        g = conductance_per_soma_area(0, 2800)
        assert g == 0

    def test_conductance_per_soma_area_zero_area(self):
        """Test conductance with zero soma area."""
        g = conductance_per_soma_area(0.3, 0)
        assert g == 0


class TestNumericalStability:
    """Test that conversions preserve numerical precision."""

    def test_conversion_precision_voltage(self):
        """Test voltage conversion precision over wide range."""
        for v_mv in [-100, -65, 0, 30, 100]:
            v_v = mV_to_V(v_mv)
            v_back = V_to_mV(v_v)
            assert abs(v_back - v_mv) < 1e-10

    def test_conversion_precision_current(self):
        """Test current conversion precision."""
        for i_pa in [-2000, -500, 0, 100, 5000]:
            i_na = pA_to_nA(i_pa)
            i_back = nA_to_pA(i_na)
            assert abs(i_back - i_pa) < 1e-10

    def test_tau_membrane_with_realistic_values(self):
        """Test tau calculation with realistic mammalian ranges."""
        # Typical ranges from literature
        R_values = [5000, 10000, 20000, 50000]  # Ω·cm²
        C_values = [0.5, 0.8, 1.0, 1.2]  # μF/cm²

        for R in R_values:
            for C in C_values:
                tau = tau_membrane_ms(R, C)
                # Mammalian neurons: tau typically 5–100 ms
                assert 1 < tau < 200, f"tau={tau} ms out of expected range for R={R}, C={C}"
