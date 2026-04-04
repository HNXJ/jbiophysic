import jax.numpy as jnp
import jaxley as jx

class Neuromodulator(jx.channels.Channel):
    """
    Base Neuromodulator class (implemented as a slow-decaying state 'm').
    release: input current or spike-driven factor.
    """
    def __init__(self, name: str, tau=1000.0):
        super().__init__(name=name)
        self.channel_params = {"tau": tau}
        self.channel_states = {"m": 0.0}

    def update_states(self, states, dt, v, params):
        # Decaying dynamics: dm/dt = -m/tau
        # We assume 'release' comes from external input or a specific synapse
        dm = -states["m"] / params["tau"]
        return {"m": states["m"] + dt * dm}

    def compute_current(self, states, v, params):
        # Modulators don't typically generate current directly, 
        # but they influence other channel parameters.
        return 0.0

class Dopamine(Neuromodulator):
    """
    Dopamine (DA): Increases NMDA gain, Decreases Leak.
    """
    def __init__(self, name="DA", tau=1500.0):
        super().__init__(name=name, tau=tau)

class Serotonin(Neuromodulator):
    """
    Serotonin (5-HT): Suppresses AMPA, Increases GABA.
    """
    def __init__(self, name="5HT", tau=2000.0):
        super().__init__(name=name, tau=tau)

class ACh(Neuromodulator):
    """
    Acetylcholine (ACh): Increases AMPA, Decreases Adaptation.
    """
    def __init__(self, name="ACh", tau=1000.0):
        super().__init__(name=name, tau=tau)

class Norepinephrine(Neuromodulator):
    """
    Norepinephrine (NE): Increases Gain / Excitability.
    """
    def __init__(self, name="NE", tau=800.0):
        super().__init__(name=name, tau=tau)

# --- Modulation Engine (Functional Axis 2) ---

def compute_modulation(state: jnp.ndarray):
    """
    Functional Modulation (Axis 2):
    - da: Precision scaling, NMDA gain boost, STDP scaling.
    - ach: Sensory input gain, Top-down suppression.
    """
    # Assuming state is indexed or a dict-like structure (e.g. from jaxley)
    da = state.get("da", 0.0)
    ach = state.get("ach", 0.0)
    
    return {
        "precision": 1.0 + da,
        "nmda_gain": 1.0 + 0.5 * da,
        "stdp_scale": 1.0 + da,
        "input_gain": 1.0 + ach,
        "topdown_gain": jnp.maximum(1.0 - 0.5 * ach, 0.0)
    }

def apply_modulation(params, mod_factors):
    """
    Functional transform of parameters (Conductance Scaling).
    - mod_factors: output of compute_modulation()
    """
    params["precision"] = mod_factors["precision"]
    params["nmda_g"] *= mod_factors["nmda_gain"]
    params["stdp_scale"] *= mod_factors["stdp_scale"]
    params["input_gain"] *= mod_factors["input_gain"]
    params["topdown_gain"] *= mod_factors["topdown_gain"]
    
    return params
