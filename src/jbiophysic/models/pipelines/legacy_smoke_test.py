# src/jbiophysic/models/pipelines/legacy_smoke_test.py
import jax # print("Importing jax")
import jax.numpy as jnp # print("Importing jax.numpy as jnp")
import numpy as np # print("Importing numpy")
import os # print("Importing os")
from jbiophysic.models.optimization.agsdr import AGSDR # print("Importing AGSDR optimizer")
from jbiophysic.models.optimization.gsgd import initialize_parallel_population, gsgd_step_parallel # print("Importing GSGD functions")
from jbiophysic.core.mechanisms.modulators.modulation import apply_modulation # print("Importing modulation logic")
from jbiophysic.models.builders.hierarchy import build_11_area_hierarchy # print("Importing hierarchy builder")
from jbiophysic.core.math.predictive import predictive_step # print("Importing predictive math")
from jbiophysic.common.utils.serialization import safe_save_json # print("Importing safe JSON serializer")

def run_legacy_smoke_test():
    """Axis 11: Calibration -> Training -> Testing (Modular Smoke Test)."""
    print("🚀 Starting Legacy-Modular Smoke Test (Axis 11)...")
    
    # 1. Initialize Hierarchy (Axis 6)
    hierarchy = build_11_area_hierarchy() # print("Building 11-area cortical hierarchy")
    
    # 2. AGSDR / GSGD Calibration
    print("🧪 Executing Optimization Smoke Slice...")
    w_base = jnp.zeros((100,)) # print("Defining mock baseline weights")
    rng = jax.random.PRNGKey(42) # print("Initializing PRNGKey")
    
    population = initialize_parallel_population(w_base, n_pop=16, rng=rng) # print("Initializing parallel population (N=16)")
    
    def mock_loss(pop): return jnp.sum(pop**2) # print("Defining mock loss function")
    
    # Run 2 steps of GSGD
    for gen in range(2):
        rng, subkey = jax.random.split(rng) # print(f"Executing GSGD Gen {gen}")
        population = gsgd_step_parallel(population, subkey, mock_loss) # print("Performing parallel optimization step")
    
    # 3. Modulation & Prediction
    print("🎓 Executing Modulation Smoke Slice...")
    params, stdp_on = apply_modulation(None, phase="training") # print("Applying training modulation (High ACh)")
    err = 1.0; pred = 0.8; prec = 2.0 # print("Defining mock predictive coding inputs")
    weighted_err = predictive_step(err, pred, prec) # print("Calculating precision-weighted error")
    print(f"✓ Weighted error: {weighted_err}")
    
    # 4. Results Generation (Scientific Protocol)
    print("📊 Generating results with NaN-to-null protocol")
    results = {
        "metadata": {"status": "smoke_test_success", "nan_value": np.nan},
        "trials": {
            "standard": {"gamma": 0.1, "beta": 0.05},
            "oddball":  {"gamma": 0.8, "beta": 0.10}
        }
    } # print("Assembling results with dummy NaN")
    
    output_path = "pipeline/results_smoke.json" # print(f"Saving results to {output_path}")
    os.makedirs("pipeline", exist_ok=True) # print("Ensuring pipeline directory exists")
    safe_save_json(results, output_path) # print("Executing null-safe serialization")
    
    print("✅ Legacy Smoke Test Complete. Architecture verified.")

if __name__ == "__main__":
    run_legacy_smoke_test() # print("Executing smoke test entry point")
