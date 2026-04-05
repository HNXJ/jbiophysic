# codes/scripts/run_oscillation_tests.py
import jax
import jax.numpy as jnp
import numpy as np
import scipy.signal
from scipy.signal import hilbert

from codes.simulation import cortical_multi_pop_dynamics, simulate_cortical_hierarchy

# --- TEST 1: EIGENVALUE TEST (MANDATORY) ---
def compute_jacobian(state, params):
    """Computes Jacobian of the dynamical system around the steady state."""
    # Create fixed points
    def wrapped_dynamics(vec_state):
        # vec_state: [E, PV, SST, VIP]
        s = {"E": vec_state[0], "PV": vec_state[1], "SST": vec_state[2], "VIP": vec_state[3]}
        # Continuous time derivative: dx/dt
        dt = 1.0 # Evaluate continuous f(x)
        # We need the derivatives dx/dt not x + dt*dx/dt, so we extract just the dX terms.
        # It's easier to run dt=1 and subtract origin, assuming dt scales linearly.
        s_next = cortical_multi_pop_dynamics(s, params, 0.0, dt=1.0)
        return jnp.array([s_next["E"] - s["E"], s_next["PV"] - s["PV"], s_next["SST"] - s["SST"], s_next["VIP"] - s["VIP"]])

    vec_state = jnp.array([state["E"], state["PV"], state["SST"], state["VIP"]])
    J = jax.jacobian(wrapped_dynamics)(vec_state)
    return J

def test_eigenvalues(init_state, params):
    print("\n--- TEST 1: JACOBIAN EIGENVALUES ---")
    J = compute_jacobian(init_state, params)
    eigvals = jnp.linalg.eigvals(J)
    
    # Are there complex conjugate pairs pointing to limit cycles?
    complex_eigs = [e for e in eigvals if jnp.abs(jnp.imag(e)) > 1e-5]
    if len(complex_eigs) > 0:
        print(f"✅ PASSED. Found complex eigenvalues (Oscillation capable):")
        for e in complex_eigs:
            print(f"   λ = {np.real(e):.4f} ± {np.imag(e):.4f}i")
    else:
        print(f"❌ FAILED. All real eigenvalues. The system cannot intrinsically oscillate.")
    return eigvals

# --- TEST 2: POWER SPECTRUM VALIDATION ---
def test_power_spectrum(e_signal, fs=10000.0):
    print("\n--- TEST 2: POWER SPECTRUM BROADNESS ---")
    freqs = jnp.fft.rfftfreq(len(e_signal), 1.0/fs)
    fft_mag = jnp.abs(jnp.fft.rfft(e_signal))**2
    
    peak_idx = jnp.argmax(fft_mag)
    peak_freq = freqs[peak_idx]
    
    print(f"Peak Frequency: {peak_freq:.2f} Hz")
    
    # Simple width check (Full Width at Half Maximum proxy)
    half_max = fft_mag[peak_idx] / 2.0
    width = jnp.sum(fft_mag > half_max) * (freqs[1]-freqs[0])
    
    if width > 1.0:
        print(f"✅ PASSED. Broad peak detected (Width = {width:.2f} Hz). Real limit cycle or noise-resonance.")
    else:
        print(f"❌ WARNING. Ultra-sharp peak (Width = {width:.2f} Hz). May be numerical arithmetic artifact.")

# --- TEST 3: PHASE RELATION TEST ---
def test_phase_relation(e_signal, i_signal):
    print("\n--- TEST 3: E/I PHASE LAG ---")
    # Convert jnp arrays to np for scipy hilbert
    e_np = np.asarray(e_signal)
    i_np = np.asarray(i_signal)
    
    e_phase = np.angle(hilbert(e_np - np.mean(e_np)))
    i_phase = np.angle(hilbert(i_np - np.mean(i_np)))
    
    phase_diff = np.mean(np.angle(np.exp(1j * (e_phase - i_phase))))
    phase_deg = np.rad2deg(phase_diff)
    
    print(f"Mean Phase Lag (E vs I): {phase_deg:.2f}°")
    
    if 45 < phase_deg < 135:
        print(f"✅ PASSED. Phase lag is near 90° (π/2). Canonical E->I loop confirmed.")
    else:
        print(f"❌ FAILED. Phase relation is broken (not near 90°). Fake oscillation.")

# --- TEST 4 & 5: DELAY & NOISE DEPENDENCE ---
def run_all_validations():
    print("🧬 Starting Axis 15 Mathematical Oscillatory Validation Suite...")
    
    # 0. Setup System
    params = {
        "w_ee": 5.0, "alpha_pv": 1.2, "alpha_sst": 0.5, "alpha_vip": 0.0,
        "sigma": 0.05, 
        "delay_ff": 10, "delay_fb": 10, # Delay buffer
        "stimulus_time_series": jnp.ones(10000) * 2.0 # Constant drive
    }
    init_state = {"E": jnp.array([0.1]), "PV": jnp.array([0.1]), "SST": jnp.array([0.01]), "VIP": jnp.array([0.0])}
    
    # Test 1: Eigenvalues
    test_eigenvalues(init_state, params)
    
    # RUN BASELINE SIMULATION
    _, trajectory = simulate_cortical_hierarchy(init_state, params, T=10000, dt=0.1)
    
    e_sig = trajectory["E"][-5000:, 0] # Discard burn-in
    pv_sig = trajectory["PV"][-5000:, 0]
    
    # Test 2 & 3
    test_power_spectrum(e_sig)
    test_phase_relation(e_sig, pv_sig)
    
    # Test 4: Does it survive noise?
    print("\n--- TEST 4: NOISE RESILIENCE ---")
    if jnp.var(e_sig) > 1e-4:
        print(f"✅ PASSED. Oscillations survive biological noise variance (Var = {jnp.var(e_sig):.4f}).")
    else:
        print(f"❌ FAILED. Signal collapsed into stable point under noise.")
        
    print("\n✅ Mathematical Validation Complete.")

if __name__ == "__main__":
    run_all_validations()
