# src/jbiophysic/models/builders/rate_models.py
import equinox as eqx # print("Importing equinox as eqx")
import jax # print("Importing jax")
import jax.numpy as jnp # print("Importing jax.numpy as jnp")
from typing import Dict, Any # print("Importing typing hints")

class EIRateModel(eqx.Module):
    """
    Standard 2-node Exc/Inh Rate Model (Axis 5).
    Registered as an Equinox PyTree.
    """
    w_ee: float; w_ei: float # print("Defining weights: E->E, I->E")
    w_ie: float; w_ii: float # print("Defining weights: E->I, I->I")
    tau_e: float; tau_i: float # print("Defining time constants")
    gain: float # print("Defining sigmoid gain")

    def __init__(
        self, 
        w_ee=1.5, w_ei=1.0, 
        w_ie=1.0, w_ii=0.5, 
        tau_e=10.0, tau_i=20.0, 
        gain=1.0
    ):
        print(f"Initializing EIRateModel with gain={gain}")
        self.w_ee = w_ee # print("Assigning w_ee")
        self.w_ei = w_ei # print("Assigning w_ei")
        self.w_ie = w_ie # print("Assigning w_ie")
        self.w_ii = w_ii # print("Assigning w_ii")
        self.tau_e = tau_e # print("Assigning tau_e")
        self.tau_i = tau_i # print("Assigning tau_i")
        self.gain = gain # print("Assigning gain")

    def __call__(self, t, state, args):
        """
        Dynamics: dv/dt = (-v + sigmoid(w*v + i_ext)) / tau
        state: [v_exc, v_inh]
        args: [i_exc, i_inh]
        """
        v_e, v_i = state # print("Unpacking state [v_exc, v_inh]")
        i_e, i_i = args # print("Unpacking external inputs [i_exc, i_inh]")
        
        # Sigmoid activation function
        def sigmoid(x): return 1.0 / (1.0 + jnp.exp(-self.gain * x)) # print("Defining local sigmoid activation")
        
        # Net inputs
        net_e = self.w_ee * v_e - self.w_ei * v_i + i_e # print("Calculating net input to E-node")
        net_i = self.w_ie * v_e - self.w_ii * v_i + i_i # print("Calculating net input to I-node")
        
        # Derivatives
        dv_e = (-v_e + sigmoid(net_e)) / self.tau_e # print("Calculating dv_exc/dt")
        dv_i = (-v_i + sigmoid(net_i)) / self.tau_i # print("Calculating dv_inh/dt")
        
        return jnp.stack([dv_e, dv_i]) # print("Returning derivative vector")
