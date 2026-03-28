import jax
import jax.numpy as jnp
import jaxley as jx
import optax

class GradedGABAb(jx.synapses.Synapse):
    """Graded slow Inhibitory Synapse (GABAb)."""
    def __init__(self, g: float = 1.0, tauD_GABAb: Optional[float] = None):
        super().__init__()
        self.synapse_params = {
            "gGABAb": g, "EGABAb": -95.0, "tauDGABAb": 200.0, "tauRGABAb": 10.0, 
            "slopeGABAb": 5.0, "V_thGABAb": -20.0
        }
        if tauD_GABAb is not None: self.synapse_params["tauDGABAb"] = tauD_GABAb
        self.synapse_states = {"sGABAb": 0.01}
    def update_states(self, states, dt, pre_v, post_v, params):
        s = states["sGABAb"]
        activation = 0.5 * (1 + jnp.tanh((pre_v - params["V_thGABAb"]) / params["slopeGABAb"]))
        d_s = (-s / params["tauDGABAb"]) + activation * ((1 - s) / params["tauRGABAb"])
        new_s = s + d_s * dt
        new_s = jnp.where(jnp.isnan(new_s) | jnp.isinf(new_s), s, new_s)
        return {"sGABAb": new_s}
    def compute_current(self, states, pre_v, post_v, params): 
        return params["gGABAb"] * states["sGABAb"] * (post_v - params["EGABAb"])

def build_net_eig(num_e: int, num_ig: int, num_il: int, seed: Optional[int] = None):
    """
    Constructs a JAXley neural network with specified numbers of excitatory and inhibitory neurons.
    If seed is None, a random seed is generated for connectivity.
    """
    if seed is None:
        seed = int(np.random.randint(0, 2**31 - 1))
    np.random.seed(seed) # Set numpy seed for connectivity
    
    comp = jx.Compartment()
    branch = jx.Branch(comp, ncomp=2)

    e_cells = [jx.Cell(branch, parents=[-1, 0]) for _ in range(num_e)]
    i_cells_g = [jx.Cell(branch, parents=[-1, 0]) for _ in range(num_ig)]
    i_cells_l = [jx.Cell(branch, parents=[-1, 0]) for _ in range(num_il)]

    for cell in e_cells + i_cells_g + i_cells_l:
        r_d_tau_diff = np.random.uniform(5.0, 50.0)
        r_amp_diff = np.clip(np.random.uniform(-0.1, 0.1), 0.0, 0.1)
        cell.insert(HH())
        cell.insert(Inoise(initial_amp_noise=r_amp_diff, initial_tau=r_d_tau_diff, initial_mean=0.0))
        cell.radius = 1.0
        cell.length = 1.0

    net = jx.Network(e_cells + i_cells_g + i_cells_l)
    
    # E -> all (AMPA)
    pre = net.cell([ik for ik in range(num_e)]).branch(0).loc(0.0)
    post = net.cell([ik for ik in range(num_e+num_ig+num_il)]).branch(1).loc(1.0)
    fully_connect(pre, post, GradedAMPA(tauD_AMPA=2.0))

    # Ig -> all (GABAa)
    pre = net.cell([ik for ik in range(num_e, num_e+num_ig)]).branch(0).loc(0.0)
    post = net.cell([ik for ik in range(num_e+num_ig+num_il)]).branch(1).loc(1.0)
    fully_connect(pre, post, GradedGABAa(g=5.0, tauD_GABAa=5.0))

    # Il -> subset of all (GABAa)
    pre_il = net.cell([ik for ik in range(num_e+num_ig, num_e+num_ig+num_il)]).branch(0).loc(0.0)
    num_posts_to_select = int((num_e+num_ig+num_il)*0.1)
    posts_pool_indices = jnp.arange(0, num_e + num_ig + num_il)
    
    # Randomize connectivity key using the main seed
    key_conn = jax.random.PRNGKey(seed)
    selected_post_indices = np.array(jax.random.choice(key_conn, posts_pool_indices, shape=(num_posts_to_select,), replace=False))
    post_il = net.cell(selected_post_indices).branch(1).loc(1.0)
    fully_connect(pre_il, post_il, GradedGABAa(g=5.0, tauD_GABAa=5.0)) # Increased to 5.0ms

    return net


def build_pyramidal_mc(seed=None):
    """Multi-compartment Pyramidal cell (Soma + Distal Dendrite)."""
    if seed is None: seed = 42
    comp_soma = jx.Compartment()
    comp_dend = jx.Compartment()
    # Simple cable: Soma (0) -> Dendrite (1)
    cell = jx.Cell([comp_soma, comp_dend], parents=[-1, 0])
    cell.radius = 1.0
    cell.length = 100.0 # Length drives axial resistance
    return cell


# --- Cell Models (Pyramidal, PV, SST, VIP) ---

def build_pyramidal_cell():
    """Regular Spiking (RS) Pyramidal Cell with adaptation."""
    comp_soma = jx.Compartment()
    comp_dend = jx.Compartment()
    cell = jx.Cell([comp_soma, comp_dend], parents=[-1, 0])
    cell.radius = 1.0
    cell.length = 100.0
    # High adaptation, standard HH
    cell.insert(jx.channels.HH())
    # Assuming Jaxley HH has gK, gNa, gl. 
    # For RS: gK is moderate, gNa is moderate.
    return cell

def build_pv_cell():
    """Fast Spiking (FS) Parvalbumin Interneuron."""
    comp_soma = jx.Compartment()
    cell = jx.Cell([comp_soma], parents=[-1])
    cell.radius = 1.0
    cell.length = 10.0
    cell.insert(jx.channels.HH())
    # FS typically has very high gNa and gK for fast repolarization
    return cell

def build_sst_cell():
    """Low-Threshold Spiking (LTS) Somatostatin Interneuron."""
    comp_soma = jx.Compartment()
    cell = jx.Cell([comp_soma], parents=[-1])
    cell.radius = 1.0
    cell.length = 10.0
    cell.insert(jx.channels.HH())
    return cell

def build_vip_cell():
    """Irregular Spiking / Bursting VIP Interneuron."""
    comp_soma = jx.Compartment()
    cell = jx.Cell([comp_soma], parents=[-1])
    cell.radius = 0.5 # Smaller size, lower capacitance
    cell.length = 10.0
    cell.insert(jx.channels.HH())
    return cell
