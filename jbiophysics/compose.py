"""
jbiophysics.compose — Fluent builder API for biophysical networks.

Usage:
    import jbiophysics as jbp
    
    net = (jbp.NetBuilder(seed=42)
        .add_population("E", n=80, cell_type="pyramidal")
        .add_population("IG", n=15, cell_type="pv")
        .connect("E", "all", synapse="AMPA", p=0.1)
        .connect("IG", "all", synapse="GABAa", p=0.4)
        .make_trainable(["gAMPA", "gGABAa"])
        .build())
"""

import jax
import jax.numpy as jnp
import jaxley as jx
import numpy as np
import optax
from jaxley.connect import fully_connect, sparse_connect
from typing import Optional, List, Dict, Any, Tuple, Union
from dataclasses import dataclass, field

from jbiophysics.core.mechanisms.models import (
    Inoise, GradedAMPA, GradedGABAa, GradedGABAb, GradedNMDA,
    build_pyramidal_cell, build_pv_cell, build_sst_cell, build_vip_cell,
    make_synapses_independent,
)
from jbiophysics.core.optimizers.optimizers import SDR, GSDR, AGSDR

# --- Cell Type Registry ---

CELL_BUILDERS = {
    "pyramidal": build_pyramidal_cell,
    "pyr": build_pyramidal_cell,
    "pv": build_pv_cell,
    "sst": build_sst_cell,
    "vip": build_vip_cell,
}

SYNAPSE_TYPES = {
    "AMPA": GradedAMPA,
    "ampa": GradedAMPA,
    "GABAa": GradedGABAa,
    "gabaa": GradedGABAa,
    "GABAb": GradedGABAb,
    "gabab": GradedGABAb,
    "NMDA": GradedNMDA,
    "nmda": GradedNMDA,
}

OPTIMIZER_METHODS = {
    "SDR": SDR,
    "GSDR": GSDR,
    "AGSDR": AGSDR,
}


@dataclass
class PopulationSpec:
    """Specification for a cell population."""
    name: str
    n: int
    cell_type: str
    noise_amp: float = 0.05
    noise_tau: float = 20.0
    cell_builder: Any = None


@dataclass
class ConnectionSpec:
    """Specification for a synaptic connection."""
    pre: str
    post: str
    synapse: str
    p: float = 0.1
    g: Optional[float] = None
    kwargs: Dict[str, Any] = field(default_factory=dict)


class NetBuilder:
    """
    Fluent builder for composable biophysical networks.
    
    Example:
        net = (NetBuilder(seed=42)
            .add_population("E", n=80, cell_type="pyramidal")
            .add_population("I", n=20, cell_type="pv")
            .connect("E", "all", synapse="AMPA", p=0.1)
            .connect("I", "all", synapse="GABAa", p=0.4)
            .make_trainable(["gAMPA", "gGABAa"])
            .build())
    """
    
    def __init__(self, seed: Optional[int] = None):
        self.seed = seed if seed is not None else int(np.random.randint(0, 2**31 - 1))
        self._populations: List[PopulationSpec] = []
        self._connections: List[ConnectionSpec] = []
        self._trainable_params: List[str] = []
        self._built_net: Optional[jx.Network] = None
        self._pop_offsets: Dict[str, Tuple[int, int]] = {}
    
    def add_population(self, name: str, n: int, cell_type: str,
                       noise_amp: float = 0.05, noise_tau: float = 20.0) -> "NetBuilder":
        """Add a named cell population."""
        if cell_type not in CELL_BUILDERS:
            raise ValueError(f"Unknown cell_type '{cell_type}'. Options: {list(CELL_BUILDERS.keys())}")
        self._populations.append(PopulationSpec(
            name=name, n=n, cell_type=cell_type,
            noise_amp=noise_amp, noise_tau=noise_tau,
            cell_builder=CELL_BUILDERS[cell_type],
        ))
        return self
    
    def connect(self, pre: str, post: str, synapse: str, p: float = 0.1,
                g: Optional[float] = None, **kwargs) -> "NetBuilder":
        """Add a synaptic connection between populations."""
        if synapse not in SYNAPSE_TYPES:
            raise ValueError(f"Unknown synapse '{synapse}'. Options: {list(SYNAPSE_TYPES.keys())}")
        self._connections.append(ConnectionSpec(
            pre=pre, post=post, synapse=synapse, p=p, g=g, kwargs=kwargs,
        ))
        return self
    
    def make_trainable(self, params: Union[str, List[str]]) -> "NetBuilder":
        """Mark parameters as independently trainable."""
        if isinstance(params, str):
            params = [params]
        self._trainable_params.extend(params)
        return self
    
    def build(self) -> jx.Network:
        """Construct the jaxley Network from the accumulated specs."""
        np.random.seed(self.seed)
        
        all_cells = []
        offset = 0
        for pop in self._populations:
            cells = []
            for i in range(pop.n):
                cell = pop.cell_builder()
                r_amp = np.clip(np.random.uniform(pop.noise_amp * 0.5, pop.noise_amp * 1.5), 0.0, 0.5)
                cell.insert(Inoise(initial_amp_noise=r_amp, initial_tau=pop.noise_tau))
                cells.append(cell)
            self._pop_offsets[pop.name] = (offset, offset + pop.n)
            all_cells.extend(cells)
            offset += pop.n
        
        net = jx.Network(all_cells)
        
        total_n = len(all_cells)
        for conn in self._connections:
            pre_start, pre_end = self._pop_offsets[conn.pre]
            pre_indices = list(range(pre_start, pre_end))
            
            if conn.post == "all":
                post_indices = list(range(total_n))
            else:
                post_start, post_end = self._pop_offsets[conn.post]
                post_indices = list(range(post_start, post_end))
            
            syn_cls = SYNAPSE_TYPES[conn.synapse]
            syn_kwargs = dict(conn.kwargs)
            if conn.g is not None:
                syn_kwargs["g"] = conn.g
            syn = syn_cls(**syn_kwargs) if syn_kwargs else syn_cls()
            
            pre_view = net.cell(pre_indices).branch(0).loc(0.0)
            post_view = net.cell(post_indices).branch(0).loc(0.0)
            sparse_connect(pre_view, post_view, syn, p=conn.p)
        
        for param in self._trainable_params:
            make_synapses_independent(net, param)
        
        self._built_net = net
        print(f"✅ Network built: {total_n} cells, {len(self._connections)} connection rules")
        return net
    
    @property
    def population_offsets(self) -> Dict[str, Tuple[int, int]]:
        """Returns {pop_name: (start_idx, end_idx)} after build()."""
        return dict(self._pop_offsets)


class OptimizerFacade:
    """
    High-level optimizer wrapper for biophysical networks.
    
    Example:
        result = (OptimizerFacade(net, method="AGSDR", lr=1e-3)
            .set_constraints(firing_rate=(1, 100), kappa_max=0.1)
            .run(epochs=200))
    """
    
    def __init__(self, net: jx.Network, method: str = "AGSDR", lr: float = 1e-3, **optimizer_kwargs):
        self.net = net
        self.method = method
        self.lr = lr
        self.optimizer_kwargs = optimizer_kwargs
        self._constraints: Dict[str, Any] = {}
        self._target_psd: Optional[jnp.ndarray] = None
    
    def set_constraints(self, firing_rate: Optional[Tuple[float, float]] = None,
                        kappa_max: Optional[float] = None) -> "OptimizerFacade":
        if firing_rate is not None:
            self._constraints["firing_rate"] = firing_rate
        if kappa_max is not None:
            self._constraints["kappa_max"] = kappa_max
        return self
    
    def set_target(self, psd_profile: Optional[jnp.ndarray] = None) -> "OptimizerFacade":
        self._target_psd = psd_profile
        return self
    
    def _build_optimizer(self) -> Any:
        inner = optax.adam(self.lr)
        method_fn = OPTIMIZER_METHODS.get(self.method)
        if method_fn is None:
            raise ValueError(f"Unknown optimizer '{self.method}'. Options: {list(OPTIMIZER_METHODS.keys())}")
        if self.method == "SDR":
            return method_fn(learning_rate=self.lr, **self.optimizer_kwargs)
        else:
            return method_fn(inner_optimizer=inner, **self.optimizer_kwargs)
    
    def run(self, epochs: int = 100, dt: float = 0.1, t_max: float = 1500.0, seed: int = 42):
        from jbiophysics.export import ResultsReport
        
        optimizer = self._build_optimizer()
        params = self.net.get_parameters()
        opt_state = optimizer.init(params)
        key = jax.random.PRNGKey(seed)
        loss_history = []
        
        self.net.delete_recordings()
        self.net.cell("all").branch(0).loc(0.0).record("v")
        
        for epoch in range(epochs):
            key, step_key = jax.random.split(key)
            traces = jx.integrate(self.net, params=params, delta_t=dt, t_max=t_max)
            
            threshold = -20.0
            spikes = (traces[:, :-1] < threshold) & (traces[:, 1:] >= threshold)
            firing_rates = jnp.sum(spikes, axis=1) / (t_max / 1000.0)
            
            fr_range = self._constraints.get("firing_rate", (1.0, 100.0))
            loss = jnp.mean(jnp.exp(fr_range[0] - firing_rates) + jnp.exp(firing_rates - fr_range[1]))
            loss_history.append(float(loss))
            
            if epoch % 10 == 0:
                print(f"  Epoch {epoch}: loss={loss:.4f}, mean_FR={jnp.mean(firing_rates):.1f} Hz")
        
        return ResultsReport(
            traces=np.array(traces),
            params=params,
            loss_history=loss_history,
            dt=dt, t_max=t_max,
            metadata={"method": self.method, "epochs": epochs, "constraints": self._constraints},
        )
