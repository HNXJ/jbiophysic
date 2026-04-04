import jax.numpy as jnp
import jaxley as jx

class graded_synapse(jx.connect.Synapse):
    """
    Compositional Synapse based on DynaSim kinetic equations.
    ds/dt = alpha * T(V_pre) * (1-s) - beta * s
    """
    def __init__(self, pre, post, name: str, alpha=1.0, beta=0.1, E=0.0):
        super().__init__(pre, post, name=name)
        self.synapse_params = {
            "g": 0.5, 
            "E": E, 
            "alpha": alpha, 
            "beta": beta,
            "T_max": 1.0,
            "V_th": 2.0,
            "V_slope": 5.0
        }
        self.synapse_states = {"s": 0.0}

    def update_states(self, states, dt, v_pre, v_post, params):
        T = params["T_max"] / (1.0 + jnp.exp(-(v_pre - params["V_th"]) / params["V_slope"]))
        ds = params["alpha"] * T * (1.0 - states["s"]) - params["beta"] * states["s"]
        return {"s": states["s"] + dt * ds}

    def compute_current(self, states, v_pre, v_post, params):
        return params["g"] * states["s"] * (v_post - params["E"])

class graded_ampa(graded_synapse):
    """Fast Excitatory (Fast rise, Fast decay)."""
    def __init__(self, pre, post, name="AMPA"):
        super().__init__(pre, post, name=name, alpha=1.1, beta=0.19, E=0.0)

class graded_gabaa(graded_synapse):
    """Fast Inhibitory."""
    def __init__(self, pre, post, name="GABAa"):
        super().__init__(pre, post, name=name, alpha=0.53, beta=0.18, E=-80.0)

class graded_gabab(graded_synapse):
    """Slow Inhibitory (Metabotropic approximation)."""
    def __init__(self, pre, post, name="GABAb"):
        super().__init__(pre, post, name=name, alpha=0.01, beta=0.003, E=-90.0)

class graded_nmda(graded_synapse):
    """Slow Excitatory + Magnesium Block."""
    def __init__(self, pre, post, name="NMDA", Mg=1.0):
        super().__init__(pre, post, name=name, alpha=0.072, beta=0.0066, E=0.0)
        self.synapse_params["Mg"] = Mg

    def voltage_factor(self, V, Mg):
        return 1.0 / (1.0 + Mg * jnp.exp(-0.062 * V) / 3.57)

    def compute_current(self, states, v_pre, v_post, params):
        g_nmda = params["g"] * self.voltage_factor(v_post, params["Mg"])
        return g_nmda * states["s"] * (v_post - params["E"])
