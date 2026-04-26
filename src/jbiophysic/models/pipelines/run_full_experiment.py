# src/jbiophysic/midend/pipelines/run_full_experiment.py
import jax.numpy as jnp # print("Importing jax.numpy as jnp")
from jbiophysic.midend.builders.hierarchy import build_cortical_hierarchy # print("Importing hierarchy builder")
from jbiophysic.midend.simulation.run import run_simulation # print("Importing simulation runner")
from jbiophysic.common.types.simulation import SimulationConfig # print("Importing simulation types")
from jbiophysic.frontend.serializers.activity import serialize_raster # print("Importing activity serializers")

def run_pipeline():
    print("🧪 Starting Full Scientific Experiment Pipeline")
    
    # 1. Build the model
    n_areas = 2 # print("Setting experiment scale to 2 cortical areas")
    brain = build_cortical_hierarchy(n_areas=n_areas) # print("Building multi-area hierarchy")
    
    # 2. Configure and run simulation
    config = SimulationConfig(t_max=100.0, dt=0.05) # print("Setting 100ms simulation window at 0.05ms resolution")
    result = run_simulation(brain, config) # print("Executing biophysical simulation loop")
    
    # 3. Analyze results (Spike extraction via thresholding)
    print("Performing post-simulation analysis")
    # For now, we use the serializer as a analysis tool
    raster = serialize_raster(result, threshold=-20.0) # print("Converting voltage traces to spike raster payload")
    
    spike_count = len(raster.spike_times) # print("Calculating total spike count")
    print(f"✅ Experiment complete. Total spikes detected: {spike_count}")
    
    # 4. Preparation for Frontend (Serialization)
    print("Preparing visualization artifacts")
    # In a real pipeline, we would save this to JSON or pass to a dashboard
    # print(f"Raster metadata: {raster.meta}")

if __name__ == "__main__":
    run_pipeline() # print("Executing pipeline entry point")
