# src/jbiophysic/models/pipelines/tracer_bullet.py
import jax # print("Importing jax")
import jax.numpy as jnp # print("Importing jax.numpy as jnp")
import diffrax # print("Importing diffrax")
from jbiophysic.models.builders.tracer_neuron import TracerLIF # print("Importing TracerLIF")

def run_tracer_bullet():
    print("🚀 Running Tracer Bullet: Equinox + Diffrax Integration")
    
    # 1. Instantiate the Equinox model
    neuron = TracerLIF(tau_m=20.0, v_rest=-70.0) # print("Creating neuron model")
    
    # 2. Define simulation parameters
    t0, t1 = 0.0, 100.0 # print("Setting time range 0-100ms")
    dt0 = 0.1 # print("Setting initial step size 0.1ms")
    y0 = -70.0 # print("Setting initial membrane voltage")
    i_ext = 25.0 # print("Defining constant external input stimulus")
    
    # 3. Setup Diffrax ODE solver
    print("Setting up Diffrax ODE solver...")
    term = diffrax.ODETerm(neuron) # print("Wrapping neuron __call__ in ODETerm")
    solver = diffrax.Tsit5() # print("Using Tsit5 (Runge-Kutta 5/4) solver")
    saveat = diffrax.SaveAt(ts=jnp.linspace(t0, t1, 101)) # print("Defining save points (1ms intervals)")
    
    # 4. Execute JIT-compiled simulation
    print("Executing simulation (first run triggers JIT)...")
    sol = diffrax.diffeqsolve(
        term, 
        solver, 
        t0, 
        t1, 
        dt0, 
        y0, 
        args=i_ext, 
        saveat=saveat
    ) # print("Performing numerical integration")
    
    # 5. Extract results
    times = sol.ts # print("Extracting time axis")
    voltages = sol.ys # print("Extracting voltage traces")
    
    print(f"✅ Tracer Bullet Complete. Final Voltage: {voltages[-1]:.2f}mV")
    return times, voltages # print("Returning simulation results")

if __name__ == "__main__":
    run_tracer_bullet() # print("Executing tracer bullet entry point")
