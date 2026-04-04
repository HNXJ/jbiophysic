import jaxley as jx

class SafeHH(jx.channels.HH):
    """
    Standard HH with explicit 'HH' name to prevent KeyErrors during 
    group-based parameter setting in Jaxley 0.13+.
    """
    def __init__(self, name: str = "HH"):
        super().__init__(name=name)

class Inoise(jx.channels.Channel):
    """
    White noise current generator as a Jaxley Channel.
    Enables per-compartment stochasticity with grad-transparency.
    """
    def __init__(self, name: str = "Inoise"):
        super().__init__(name=name)
        self.channel_params = {}
        self.channel_states = {"noise": 0.0}

    def update_states(self, states, dt, v, params):
        key = jx.random.PRNGKey(0) # Logic should vary key via simulation loop
        noise = jx.random.normal(key, shape=v.shape)
        return {"noise": noise}

    def compute_current(self, states, v, params):
        return states["noise"]
