# src/jbiophysic/models/builders/tracer_neuron.py
from jbiophysic.common.utils.logging import get_logger

logger = get_logger(__name__)

import equinox as eqx
import jax.numpy as jnp
from typing import Dict, Any

class TracerLIF(eqx.Module):
    """
    Tracer Bullet: A minimal Leaky Integrate-and-Fire model.
    Registered as an Equinox PyTree for JAX compatibility.
    """
    tau_m: float
    v_rest: float
    v_thresh: float
    r_m: float

    def __init__(self, tau_m=20.0, v_rest=-70.0, v_thresh=-50.0, r_m=1.0):
        logger.info(f"Initializing TracerLIF: tau={tau_m}, v_rest={v_rest}")
        self.tau_m = tau_m
        self.v_rest = v_rest
        self.v_thresh = v_thresh
        self.r_m = r_m

    def __call__(self, t, v, args):
        """
        Dynamics function for ODE solver.
        dv/dt = (-(v - v_rest) + r_m * I_ext) / tau_m
        """
        # Note: args contains external input I_ext
        i_ext = args
        dv_dt = (-(v - self.v_rest) + self.r_m * i_ext) / self.tau_m
        return dv_dt
