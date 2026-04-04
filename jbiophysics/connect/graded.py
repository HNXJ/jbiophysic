import jax.numpy as jnp
import jaxley as jx

class _GradedSynapse(jx.connect.Synapse):
    """
    Base class for graded synaptic transmission.
    Prevents code duplication across AMPA/GABA/NMDA variants.
    """
    def __init__(self, pre, post, name: str):
        super().__init__(pre, post, name=name)
        self.synapse_params = {"g": 1.0, "reverse_potential": 0.0}
        self.synapse_states = {"s": 0.0}

    def update_states(self, states, dt, v_pre, v_post, params):
        # Sigmoidal activation logic
        s_inf = 1.0 / (1.0 + jnp.exp(-(v_pre + 10.0) / 2.0))
        tau = 2.0
        ds = (s_inf - states["s"]) / tau
        return {"s": states["s"] + dt * ds}

    def compute_current(self, states, v_pre, v_post, params):
        return params["g"] * states["s"] * (v_post - params["reverse_potential"])

class GradedAMPA(_GradedSynapse):
    def __init__(self, pre, post, name="AMPA"):
        super().__init__(pre, post, name=name)
        self.synapse_params["reverse_potential"] = 0.0

class GradedGABAa(_GradedSynapse):
    def __init__(self, pre, post, name="GABAa"):
        super().__init__(pre, post, name=name)
        self.synapse_params["reverse_potential"] = -80.0

class GradedGABAb(_GradedSynapse):
    def __init__(self, pre, post, name="GABAb"):
        super().__init__(pre, post, name=name)
        self.synapse_params["reverse_potential"] = -90.0

class GradedNMDA(_GradedSynapse):
    def __init__(self, pre, post, name="NMDA"):
        super().__init__(pre, post, name=name)
        self.synapse_params["reverse_potential"] = 0.0
