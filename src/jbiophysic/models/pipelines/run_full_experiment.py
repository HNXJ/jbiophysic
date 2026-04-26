# src/jbiophysic/models/pipelines/run_full_experiment.py
import jax.numpy as jnp
from jbiophysic.models.builders.hierarchy import build_cortical_hierarchy
from jbiophysic.models.simulation.run import run_simulation
from jbiophysic.common.types.simulation import SimulationConfig
from jbiophysic.viz.serializers.activity import serialize_raster

def run_pipeline():
    print("🧪 Starting Full Scientific Experiment Pipeline")
    
    # 1. Build the model
    n_areas = 2
    brain = build_cortical_hierarchy(n_areas=n_areas)
    
    # 2. Configure and run simulation
    config = SimulationConfig(t_max=100.0, dt=0.05)
    result = run_simulation(brain, config)
    
    # 3. Analyze results (Spike extraction via thresholding)
    print("Performing post-simulation analysis")
    # For now, we use the serializer as a analysis tool
    raster = serialize_raster(result, threshold=-20.0)
    
    spike_count = len(raster.spike_times)
    print(f"✅ Experiment complete. Total spikes detected: {spike_count}")
    
    # 4. Preparation for Frontend (Serialization)
    print("Preparing visualization artifacts")
    # In a real pipeline, we would save this to JSON or pass to a dashboard
   

if __name__ == "__main__":
    run_pipeline()
