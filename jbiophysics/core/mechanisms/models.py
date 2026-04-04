import jax
import jax.numpy as jnp
import numpy as np
import jaxley as jx
from jaxley.channels import HH, Leak
from jaxley.synapses import Synapse
from jaxley.connect import fully_connect
from typing import Optional

class SafeHH(jx.channels.HH):
    """Hodgkin-Huxley with physical limits and NaN guards."""
    def __init__(self, name: str = "HH"):
        super().__init__(name)

    def update_states(self, states, dt, v, params):
        # Clip voltage to physiological range before HH calculation
        v_safe = jnp.clip(v, -100.0, 100.0)
        new_states = super().update_states(states, dt, v_safe, params)
        # Apply NaN guard and physical limits to gating variables [0, 1]
        for k, val in new_states.items():
            safe_val = jnp.nan_to_num(val, nan=0.0, posinf=1.0, neginf=0.0)
            new_states[k] = jnp.clip(safe_val, 0.0, 1.0)
        return new_states

    def compute_current(self, states, v, params):
        v_safe = jnp.clip(v, -100.0, 100.0)
        return super().compute_current(states, v_safe, params)

class Inoise(jx.channels.Channel):
    """Stochastic Ornstein-Uhlenbeck noise channel."""
    def __init__(self, name: str = "Inoise", initial_seed: Optional[int] = None, 
                 initial_amp_noise: float = 0.01, initial_tau: float = 20.0, 
                 initial_mean: float = 0.0):
        self.current_is_in_mA_per_cm2 = True
        super().__init__(name)
        
        seed = float(initial_seed) if initial_seed is not None else float(np.random.randint(0, 2**16 - 1))
        
        self.channel_params = {
            "amp_noise": float(initial_amp_noise), 
            "mean": float(initial_mean), 
            "tau": float(initial_tau),
            "seed": seed
        }
        self.channel_states = {"n": 0.00, "step": 0.0}
        self.current_name = "i_noise"

    def update_states(self, states, dt, v, params):
        n, step, seeds_int = states["n"], states["step"], params["seed"].astype(int)
        
        # Consistent key generation for VMAP or single
        if seeds_int.ndim == 0:
            base_key = jax.random.PRNGKey(seeds_int)
            step_key = jax.random.fold_in(base_key, step.astype(int))
            xi = jax.random.normal(step_key)
        else:
            base_key = jax.vmap(jax.random.PRNGKey)(seeds_int)
            step_key = jax.vmap(jax.random.fold_in)(base_key, step.astype(int))
            xi = jax.vmap(jax.random.normal)(step_key)
            
        mu, sigma, tau = params["mean"], params["amp_noise"], params["tau"]
        drift = (mu - n) / tau * dt
        diffusion = sigma * jnp.sqrt(2.0 / tau) * xi * jnp.sqrt(dt)
        new_n = n + drift + diffusion
        new_n = jnp.nan_to_num(new_n, nan=0.0)
        return {"n": new_n, "step": step + 1.0}

    def compute_current(self, states, v, params): return -states["n"]
    def init_state(self, states, v, params, delta_t): return {"n": params["mean"], "step": 0.0}

class _GradedSynapse(jx.synapses.Synapse):
    """Base class for graded synapses to reduce duplication."""
    def __init__(self, suffix: str, g: float, e: float, tauD: float, tauR: float, V_th: float, slope: float):
        super().__init__()
        self.suffix = suffix
        self.synapse_params = {
            f"g{suffix}": g, f"E{suffix}": e, f"tauD{suffix}": tauD, 
            f"tauR{suffix}": tauR, f"V_th{suffix}": V_th, f"slope{suffix}": slope
        }
        self.synapse_states = {f"s{suffix}": 0.1}

    def update_states(self, states, dt, pre_v, post_v, params):
        s = states[f"s{self.suffix}"]
        activation = 0.5 * (1 + jnp.tanh((pre_v - params[f"V_th{self.suffix}"]) / params[f"slope{self.suffix}"]))
        d_s = (-s / params[f"tauD{self.suffix}"]) + activation * ((1 - s) / params[f"tauR{self.suffix}"])
        new_s = jnp.clip(jnp.nan_to_num(s + d_s * dt, nan=0.0), 0.0, 1.0)
        return {f"s{self.suffix}": new_s}

    def compute_current(self, states, pre_v, post_v, params):
        return params[f"g{self.suffix}"] * states[f"s{self.suffix}"] * (post_v - params[f"E{self.suffix}"])

class GradedAMPA(_GradedSynapse):
    def __init__(self, g: float = 2.5, tauD_AMPA: float = 5.0):
        super().__init__("AMPA", g, 0.0, tauD_AMPA, 0.2, -20.0, 5.0)

class GradedGABAa(_GradedSynapse):
    def __init__(self, g: float = 5.0, tauD_GABAa: float = 5.0):
        super().__init__("GABAa", g, -80.0, tauD_GABAa, 0.5, -20.0, 5.0)

class GradedGABAb(_GradedSynapse):
    def __init__(self, g: float = 1.0, tauD_GABAb: float = 200.0):
        super().__init__("GABAb", g, -95.0, tauD_GABAb, 10.0, -20.0, 5.0)

class GradedNMDA(_GradedSynapse):
    def __init__(self, g: float = 1.0, tauD_NMDA: float = 100.0):
        super().__init__("NMDA", g, 0.0, tauD_NMDA, 2.0, -20.0, 5.0)
        self.synapse_params["Mg"] = 1.0

    def compute_current(self, states, pre_v, post_v, params):
        m_block = 1.0 / (1.0 + 0.28 * jnp.exp(-0.062 * post_v))
        return params["gNMDA"] * states["sNMDA"] * m_block * (post_v - params["ENMDA"])

def build_net_eig(num_e: int, num_ig: int, num_il: int, seed: Optional[int] = None):
    if seed is None:
        seed = int(np.random.randint(0, 2**31 - 1))
    np.random.seed(seed)
    
    comp = jx.Compartment()
    branch = jx.Branch(comp, ncomp=2)

    e_cells = [jx.Cell(branch, parents=[-1, 0]) for _ in range(num_e)]
    i_cells_g = [jx.Cell(branch, parents=[-1, 0]) for _ in range(num_ig)]
    i_cells_l = [jx.Cell(branch, parents=[-1, 0]) for _ in range(num_il)]

    for cell in e_cells + i_cells_g + i_cells_l:
        r_d_tau_diff = np.random.uniform(5.0, 50.0)
        r_amp_diff = np.clip(np.random.uniform(-0.1, 0.1), 0.0, 0.1)
        cell.insert(SafeHH(name="HH"))
        cell.insert(Inoise(initial_amp_noise=r_amp_diff, initial_tau=r_d_tau_diff, initial_mean=0.0))
        cell.radius = 1.0
        cell.length = 1.0

    net = jx.Network(e_cells + i_cells_g + i_cells_l)
    
    pre = net.cell([ik for ik in range(num_e)]).branch(0).loc(0.0)
    post = net.cell([ik for ik in range(num_e+num_ig+num_il)]).branch(1).loc(1.0)
    fully_connect(pre, post, GradedAMPA(tauD_AMPA=2.0))

    pre = net.cell([ik for ik in range(num_e, num_e+num_ig)]).branch(0).loc(0.0)
    post = net.cell([ik for ik in range(num_e+num_ig+num_il)]).branch(1).loc(1.0)
    fully_connect(pre, post, GradedGABAa(g=5.0, tauD_GABAa=5.0))

    pre_il = net.cell([ik for ik in range(num_e+num_ig, num_e+num_ig+num_il)]).branch(0).loc(0.0)
    num_posts_to_select = int((num_e+num_ig+num_il)*0.1)
    posts_pool_indices = jnp.arange(0, num_e + num_ig + num_il)
    key_conn = jax.random.PRNGKey(seed)
    selected_post_indices = np.array(jax.random.choice(key_conn, posts_pool_indices, shape=(num_posts_to_select,), replace=False))
    post_il = net.cell(selected_post_indices).branch(1).loc(1.0)
    fully_connect(pre_il, post_il, GradedGABAa(g=5.0, tauD_GABAa=5.0))

    return net

def build_pyramidal_cell():
    comp_soma = jx.Compartment()
    comp_dend = jx.Compartment()
    cell = jx.Cell([comp_soma, comp_dend], parents=[-1, 0])
    cell.radius = 1.0
    cell.length = 100.0
    cell.insert(SafeHH(name="HH"))
    return cell

def build_pv_cell():
    comp_soma = jx.Compartment()
    cell = jx.Cell([comp_soma], parents=[-1])
    cell.radius = 1.0
    cell.length = 10.0
    cell.insert(SafeHH(name="HH"))
    return cell

def build_sst_cell():
    comp_soma = jx.Compartment()
    cell = jx.Cell([comp_soma], parents=[-1])
    cell.radius = 1.0
    cell.length = 10.0
    cell.insert(SafeHH(name="HH"))
    return cell

def build_vip_cell():
    comp_soma = jx.Compartment()
    cell = jx.Cell([comp_soma], parents=[-1])
    cell.radius = 0.5
    cell.length = 10.0
    cell.insert(SafeHH(name="HH"))
    return cell

def make_synapses_independent(net: jx.Network, param_name: str):
    net.select(edges="all").make_trainable(param_name)

def get_parameter_summary(net: jx.Network):
    params = net.get_parameters()
    summary = []
    total_elements = 0
    for i, group in enumerate(params):
        for key, value in group.items():
            size = jnp.size(value)
            total_elements += size
            summary.append({"group_index": i, "parameter": key, "count": size, "is_independent": "Yes" if size > 1 else "No (Shared)"})
    return summary, total_elements
