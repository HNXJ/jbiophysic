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
    "ampa": GradedAMPA,
    "gabaa": GradedGABAa,
    "gabab": GradedGABAb,
    "nmda": GradedNMDA,
}

OPTIMIZER_METHODS = {
    "sdr": SDR,
    "gsdr": GSDR,
    "agsdr": AGSDR,
}


@dataclass
class PopulationSpec:
    """Specification for a cell population."""
    name: str
    n: int
    cell_type: str
    noise_amp: float = 0.05
    noise_tau: float = 20.0
    noise_mean: float = 0.0
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
    Supports multi-area hierarchies.
    
    Example:
        net = (NetBuilder(seed=42)
            .add_population("E", n=80, cell_type="pyramidal", area="V1")
            .add_population("I", n=20, cell_type="pv", area="V1")
            .add_population("E", n=50, cell_type="pyramidal", area="HO")
            .connect("E", "all", synapse="AMPA", p=0.1, area="V1")
            .connect("V1.E", "HO.E", synapse="AMPA", p=0.2)
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
        self._areas: Dict[str, List[str]] = {} # area_name -> [pop_names]
    
    def add_population(self, name: str, n: int, cell_type: str,
                       noise_amp: float = 0.05, noise_tau: float = 20.0,
                       noise_mean: float = 0.0, area: str = "default") -> "NetBuilder":
        """Add a named cell population to a specific area."""
        if cell_type not in CELL_BUILDERS:
            raise ValueError(f"Unknown cell_type '{cell_type}'. Options: {list(CELL_BUILDERS.keys())}")
        
        full_name = f"{area}.{name}" if area != "default" else name
        self._populations.append(PopulationSpec(
            name=full_name, n=n, cell_type=cell_type,
            noise_amp=noise_amp, noise_tau=noise_tau,
            noise_mean=noise_mean,
            cell_builder=CELL_BUILDERS[cell_type],
        ))
        
        if area not in self._areas:
            self._areas[area] = []
        self._areas[area].append(full_name)
        
        return self
    
    def connect(self, pre: str, post: str, synapse: str, p: float = 0.1,
                g: Optional[float] = None, area: Optional[str] = None, **kwargs) -> "NetBuilder":
        """
        Add a synaptic connection between populations.
        Names can be 'PopName' or 'Area.PopName'.
        If 'area' is provided, it's prepended to names if they don't have one.
        """
        def finalize_name(name):
            if area and "." not in name and name != "all":
                return f"{area}.{name}"
            return name

        self._connections.append(ConnectionSpec(
            pre=finalize_name(pre), post=finalize_name(post), 
            synapse=synapse.lower(), p=p, g=g, kwargs=kwargs,
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
                cell.insert(Inoise(initial_amp_noise=r_amp, initial_tau=pop.noise_tau, initial_mean=pop.noise_mean))
                cells.append(cell)
            self._pop_offsets[pop.name] = (offset, offset + pop.n)
            all_cells.extend(cells)
            offset += pop.n
        
        net = jx.Network(all_cells)
        
        total_n = len(all_cells)
        for conn in self._connections:
            if conn.pre not in self._pop_offsets:
                raise ValueError(f"Source population '{conn.pre}' not found. Did you forget the area prefix?")
            
            pre_start, pre_end = self._pop_offsets[conn.pre]
            pre_indices = list(range(pre_start, pre_end))
            
            if conn.post == "all":
                post_indices = list(range(total_n))
            else:
                if conn.post not in self._pop_offsets:
                     raise ValueError(f"Target population '{conn.post}' not found.")
                post_start, post_end = self._pop_offsets[conn.post]
                post_indices = list(range(post_start, post_end))
            
            syn_cls = SYNAPSE_TYPES[conn.synapse.lower()]
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
        """Returns {pop_full_name: (start_idx, end_idx)} after build()."""
        return dict(self._pop_offsets)

    @property
    def area_offsets(self) -> Dict[str, List[Tuple[int, int]]]:
        """Returns {area_name: [(start, end), ...]}."""
        result = {}
        for area, pops in self._areas.items():
            result[area] = [self._pop_offsets[p] for p in pops]
        return result


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
        self.method = method.lower()
        self.lr = lr
        self.optimizer_kwargs = optimizer_kwargs
        self._constraints: Dict[str, Any] = {}
        self._pop_constraints: Dict[str, Dict[str, Any]] = {}
        self._lag_constraints: Dict[str, Dict[str, float]] = {}
        self._target_psd: Optional[jnp.ndarray] = None
        self._pop_offsets: Optional[Dict[str, Tuple[int, int]]] = None
    
    def set_pop_offsets(self, offsets: Dict[str, Tuple[int, int]]) -> "OptimizerFacade":
        """Link population metadata for per-area/per-population metrics."""
        self._pop_offsets = offsets
        return self

    def set_constraints(self, firing_rate: Optional[Tuple[float, float]] = None,
                        kappa_max: Optional[float] = None) -> "OptimizerFacade":
        """Set global constraints."""
        if firing_rate is not None:
            self._constraints["firing_rate"] = firing_rate
        if kappa_max is not None:
            self._constraints["kappa_max"] = kappa_max
        return self
    
    def set_pop_constraints(self, pop_name: str, firing_rate: Optional[Tuple[float, float]] = None,
                           kappa_max: Optional[float] = None) -> "OptimizerFacade":
        """Set constraints for a specific population or area."""
        if pop_name not in self._pop_constraints:
            self._pop_constraints[pop_name] = {}
        if firing_rate is not None:
            self._pop_constraints[pop_name]["firing_rate"] = firing_rate
        if kappa_max is not None:
            self._pop_constraints[pop_name]["kappa_max"] = kappa_max
        return self

    def set_lag_constraint(self, pop_name: str, target_ms: float, 
                           stimulus_onset_ms: float) -> "OptimizerFacade":
        """Set a target peak lag constraint for a population."""
        self._lag_constraints[pop_name] = {
            "target": target_ms,
            "onset": stimulus_onset_ms
        }
        return self

    def set_target(self, psd_profile: Optional[jnp.ndarray] = None) -> "OptimizerFacade":
        self._target_psd = psd_profile
        return self
    
    def _build_optimizer(self) -> Any:
        inner = optax.adam(self.lr)
        method_fn = OPTIMIZER_METHODS.get(self.method)
        if method_fn is None:
            raise ValueError(f"Unknown optimizer '{self.method}'. Options: {list(OPTIMIZER_METHODS.keys())}")
        
        if self.method == "sdr":
            return method_fn(learning_rate=self.lr, **self.optimizer_kwargs)
        else:
            return method_fn(inner_optimizer=inner, **self.optimizer_kwargs)
    
    def run(self, epochs: int = 100, dt: float = 0.1, t_max: float = 1500.0, seed: int = 42):
        from jbiophysics.export import ResultsReport
        from jbiophysics.viz.psd import compute_psd
        from jbiophysics.core.optimizers.optimizers import compute_kappa
        import gc

        optimizer = self._build_optimizer()
        params = self.net.get_parameters()
        opt_state = optimizer.init(params)
        key = jax.random.PRNGKey(seed)
        
        history = {
            "loss": [], "firing_rate": [], "kappa": [], "param_change": []
        }
        
        # Ensure recordings are set
        self.net.delete_recordings()
        self.net.cell("all").branch(0).loc(0.0).record("v")
        
        # Define loss function
        def loss_fn(p):
            tr = jx.integrate(self.net, params=p, delta_t=dt, t_max=t_max)
            tr = jnp.nan_to_num(tr, nan=0.0, posinf=100.0, neginf=-100.0)
            
            # Spike detection
            threshold = -20.0
            spikes = (tr[:, :-1] < threshold) & (tr[:, 1:] >= threshold)
            
            total_loss = 0.0
            
            def soft_range_loss(val, range_tuple):
                low, high = range_tuple
                return jnp.square(jax.nn.relu(low - val)) + jnp.square(jax.nn.relu(val - high))

            # 1. Global Constraints
            fr_global = jnp.sum(spikes) / (t_max / 1000.0) / (len(tr) + 1e-12)
            fr_range_global = self._constraints.get("firing_rate", (1.0, 100.0))
            loss_fr_global = soft_range_loss(fr_global, fr_range_global)
            
            kappa_global = compute_kappa(spikes, fs=1000.0/dt)
            kappa_max_global = self._constraints.get("kappa_max", 1.0)
            loss_kappa_global = jnp.square(jax.nn.relu(kappa_global - kappa_max_global))
            
            total_loss += loss_fr_global + 100.0 * loss_kappa_global

            # 2. Population/Area Constraints
            if self._pop_offsets:
                for pop_name, (start, end) in self._pop_offsets.items():
                    pop_spikes = spikes[start:end, :]
                    
                    if pop_name in self._pop_constraints:
                        constraints = self._pop_constraints[pop_name]
                        
                        # FR
                        if "firing_rate" in constraints:
                            fr_pop = jnp.sum(pop_spikes) / (t_max / 1000.0) / (end - start + 1e-12)
                            total_loss += soft_range_loss(fr_pop, constraints["firing_rate"])
                        
                        # Kappa
                        if "kappa_max" in constraints:
                            k_pop = compute_kappa(pop_spikes, fs=1000.0/dt)
                            total_loss += 100.0 * jnp.square(jax.nn.relu(k_pop - constraints["kappa_max"]))
                    
                    if pop_name in self._lag_constraints:
                        lag_spec = self._lag_constraints[pop_name]
                        # Soft-argmax lag estimation
                        times = jnp.arange(spikes.shape[1]) * dt
                        pop_avg = jnp.mean(pop_spikes, axis=0)
                        # Mask to window after onset (e.g., 200ms window)
                        mask = (times > lag_spec["onset"]) & (times < lag_spec["onset"] + 200.0)
                        fr_masked = pop_avg * mask
                        peak_time = jnp.sum(times * fr_masked) / (jnp.sum(fr_masked) + 1e-12)
                        actual_lag = peak_time - lag_spec["onset"]
                        # Penalize deviation from target lag
                        total_loss += 10.0 * jnp.square(actual_lag - lag_spec["target"])

            # 3. PSD Loss
            loss_psd = 0.0
            if self._target_psd is not None:
                # Approximate LFP as mean membrane potential of all cells
                lfp = jnp.mean(tr, axis=0)
                _, psd = compute_psd(lfp, dt)
                # MSE on PSD log scale
                loss_psd = jnp.mean(jnp.square(jnp.log(psd + 1e-6) - jnp.log(self._target_psd + 1e-6)))
                total_loss += 10.0 * loss_psd
            
            return total_loss, (fr_global, kappa_global)

        # JIT the step function
        grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
        
        @jax.jit
        def train_step(p, state, k):
            (loss_val, aux), grads = grad_fn(p)
            k, subkey = jax.random.split(k)
            # Handle GSDR requirements (value=loss, key=key)
            if self.method in ["gsdr", "agsdr"]:
                updates, next_state = optimizer.update(grads, state, p, value=loss_val, key=subkey)
            else:
                updates, next_state = optimizer.update(grads, state, p)
            
            new_p = optax.apply_updates(p, updates)
            return new_p, next_state, loss_val, aux, k

        print(f"🚀 OptimizerFacade ({self.method}): Running {epochs} epochs...")
        prev_p = params
        
        for epoch in range(epochs):
            params, opt_state, loss_val, (fr, kappa), key = train_step(params, opt_state, key)
            
            # Track Param Change
            flat_new, _ = jax.flatten_util.ravel_pytree(params)
            flat_old, _ = jax.flatten_util.ravel_pytree(prev_p)
            change = jnp.linalg.norm(flat_new - flat_old) / (jnp.linalg.norm(flat_old) + 1e-8)
            prev_p = params
            
            history["loss"].append(float(loss_val))
            history["firing_rate"].append(float(fr))
            history["kappa"].append(float(kappa))
            history["param_change"].append(float(change))
            
            if epoch % 10 == 0 or epoch == epochs - 1:
                print(f"  Epoch {epoch:3d} | Loss: {loss_val:.4f} | FR: {fr:.2f}Hz | Kappa: {kappa:.3f} | ΔP: {change:.2e}")
            
            if epoch % 20 == 0:
                gc.collect()

        # Final simulation for results
        final_traces = jx.integrate(self.net, params=params, delta_t=dt, t_max=t_max)
        
        return ResultsReport(
            traces=np.array(final_traces),
            params=params,
            loss_history=history["loss"],
            dt=dt, t_max=t_max,
            metadata={
                "method": self.method, 
                "epochs": epochs, 
                "constraints": self._constraints,
                "pop_constraints": self._pop_constraints,
                "history": history
            },
        )
