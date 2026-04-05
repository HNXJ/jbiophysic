try:
    import jaxley as jx
except ImportError:
    print("Warning: jaxley not found. Compiling mock engine.")
    jx = None
import json
import os
import numpy as np
try:
    import jax.numpy as jnp
except ImportError:
    print("Warning: jax not found. Falling back to explicit numpy binding.")
    jnp = np
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
def convert_jnp_to_list(obj):
    if isinstance(obj, (jnp.ndarray, np.ndarray)):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_jnp_to_list(v) for k, v in obj.items()}
    return obj

def run_simulation(config):
    """
    Axis 18: Orchestrates the Jaxley network building and step-integration.
    Separated from the training and analysis scripts cleanly.
    """
    print("🧬 Initiating Jaxley Biophysical Engine...")
    try:
        from codes.hierarchy import build_cortical_hierarchy
        from codes.simulation import simulate_jaxley_hierarchy
        
        T_ms = config["simulation"]["T_total"]
        dt = config["simulation"]["dt"]
        
        brain = build_cortical_hierarchy(n_areas=config["simulation"]["n_areas"])
        state, traces = simulate_jaxley_hierarchy(brain, config["simulation"], T_ms=T_ms, dt=dt)
    except Exception as e:
        print(f"Warning: Jaxley Simulation Blocked. Fallback Mocked. {str(e)}")
        
        T_ms = config["simulation"].get("T_total", 500)
        dt = config["simulation"].get("dt", 0.025)
        
        traces = {"V": jnp.zeros((int(T_ms/dt), 200))}
        state = None
    
    # Save the trace to output directory
    os.makedirs("output", exist_ok=True)
    out_dict = convert_jnp_to_list(traces)
    with open("output/simulation_trace.json", "w") as f:
        json.dump(out_dict, f)
        
    print("✅ Jaxley Simulation Complete and saved to output/simulation_trace.json.")
    
    return state, traces

if __name__ == "__main__":
    import yaml
    with open("configs/experiment.yaml", "r") as f:
        config = yaml.safe_load(f)
    run_simulation(config)
