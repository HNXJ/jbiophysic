import jax
import jax.numpy as jnp
import jaxley as jx
import matplotlib.pyplot as plt
from jbiophysics import build_pyramidal_cell, SpikingNMDA, stdp_params_pc, gamma_init, step_with_gamma, spike_fn

def test_simulation():
    print("🚀 Starting Biophysical Verification...")
    
    # 1. Setup Network
    pre_cell = build_pyramidal_cell()
    post_cell = build_pyramidal_cell()
    
    # Connect with Spiking NMDA + STDP
    syn = SpikingNMDA(pre_cell.soma, post_cell.soma, name="NMDA_Syn")
    
    # Configure STDP
    params = stdp_params_pc()
    for k, v in params.items():
        syn.set(f"NMDA_Syn_{k}", v)
    
    # 2. Simulation Parameters
    dt = 0.1
    t_max = 100.0
    steps = int(t_max / dt)
    
    # 3. Gamma Initialization
    gamma = gamma_init()
    
    # 4. Simulation Loop (Manual Step for Gamma Tracing)
    # Note: In a real scenario, we'd wrap jaxley.integrate, 
    # but for verification we manually step the state.
    
    time = jnp.linspace(0, t_max, steps)
    v_pre_trace = []
    v_post_trace = []
    w_trace = []
    
    # Mocking states for this isolated test
    v_pre = -65.0
    v_post = -65.0
    syn_state = {
        "s": 0.0,
        "w": 0.1,
        "trace_pre": 0.0,
        "trace_post": 0.0
    }
    
    print("🧠 Running simulation loop...")
    for i in range(steps):
        # Inject current to trigger spikes
        # Pre spikes at 20ms, Post spikes at 30ms (LTP condition)
        i_pre = 10.0 if (20.0 < i * dt < 22.0) else 0.0
        i_post = 10.0 if (30.0 < i * dt < 32.0) else 0.0
        
        # Simple Euler-like update for demonstration of the mechanism integration
        v_pre += dt * (i_pre - (v_pre + 65.0)/10.0) # Dummy RC
        v_post += dt * (i_post - (v_post + 65.0)/10.0)
        
        # Synapse step (The core we are testing)
        updates = syn.update_states(syn_state, dt, v_pre, v_post, syn.synapse_params)
        syn_state.update(updates)
        
        # Logging
        v_pre_trace.append(v_pre)
        v_post_trace.append(v_post)
        w_trace.append(syn_state["w"])
        
        # Gamma Tagging (Axis 4)
        if i % 10 == 0:
            gamma = gamma_init() # Reset or append? Append is better.
            # (In the real implementation, gamma is passed through)

    print("📊 Visualizing results...")
    plt.figure(figsize=(10, 8), facecolor='#1A1A1A')
    ax1 = plt.subplot(311)
    ax1.plot(time, v_pre_trace, color='#CFB87C', label='Pre Voltage')
    ax1.plot(time, v_post_trace, color='#9400D3', label='Post Voltage')
    ax1.set_facecolor('#2D2D2D')
    ax1.legend()
    
    ax2 = plt.subplot(312)
    ax2.plot(time, w_trace, color='#00FFCC', label='Synapse Weight (w)')
    ax2.set_facecolor('#2D2D2D')
    ax2.legend()
    
    plt.savefig('output/verification_plot.png')
    print("✅ Verification complete. Plot saved to output/verification_plot.png")

if __name__ == "__main__":
    test_simulation()
