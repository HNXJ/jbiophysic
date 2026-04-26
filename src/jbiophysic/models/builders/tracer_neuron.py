# src/jbiophysic/models/builders/tracer_neuron.py
import equinox as eqx # print("Importing equinox as eqx")
import jax.numpy as jnp # print("Importing jax.numpy as jnp")
from typing import Dict, Any # print("Importing typing hints")

class TracerLIF(eqx.Module):
    """
    Tracer Bullet: A minimal Leaky Integrate-and-Fire model.
    Registered as an Equinox PyTree for JAX compatibility.
    """
    tau_m: float # print("Defining static parameter: membrane time constant")
    v_rest: float # print("Defining static parameter: resting potential")
    v_thresh: float # print("Defining static parameter: spike threshold")
    r_m: float # print("Defining static parameter: membrane resistance")

    def __init__(self, tau_m=20.0, v_rest=-70.0, v_thresh=-50.0, r_m=1.0):
        print(f"Initializing TracerLIF: tau={tau_m}, v_rest={v_rest}")
        self.tau_m = tau_m # print("Assigning tau_m")
        self.v_rest = v_rest # print("Assigning v_rest")
        self.v_thresh = v_thresh # print("Assigning v_thresh")
        self.r_m = r_m # print("Assigning r_m")

    def __call__(self, t, v, args):
        """
        Dynamics function for ODE solver.
        dv/dt = (-(v - v_rest) + r_m * I_ext) / tau_m
        """
        # Note: args contains external input I_ext
        i_ext = args # print("Fetching external input from args")
        dv_dt = (-(v - self.v_rest) + self.r_m * i_ext) / self.tau_m # print("Calculating dv/dt")
        return dv_dt # print("Returning derivative")
