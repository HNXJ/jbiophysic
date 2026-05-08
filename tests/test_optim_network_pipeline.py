import jax
import jax.numpy as jnp

from jbiophysic.networks.builders import make_ei_network
from jbiophysic.networks.connectivity import erdos_renyi_mask, weight_matrix
from jbiophysic.optim.agsdr import adapt_alpha
from jbiophysic.optim.bounds import Bound, positive_softplus, sigmoid_bounded
from jbiophysic.optim.gsdr import gsdr_direction
from jbiophysic.optim.gsgd import gsgd_step
from jbiophysic.optim.manifests import OptimizerManifest
from jbiophysic.optim.sdr import supervised_delta_direction
from jbiophysic.pipelines.simulate import run_izhikevich_constant_current


def test_network_builder_and_connectivity():
    spec = make_ei_network(4, 2)
    assert spec.n_neurons == 6
    key = jax.random.PRNGKey(0)
    mask = erdos_renyi_mask(key, 4, 4, 1.0)
    W = weight_matrix(mask, 0.5)
    assert W.shape == (4, 4)
    assert float(jnp.trace(W)) == 0.0


def test_optimizer_helpers():
    theta = jnp.array([[0.0, 1.0], [0.5, 0.8], [1.0, 0.6]])
    loss = jnp.array([3.0, 2.0, 1.0])
    d = supervised_delta_direction(theta, loss)
    assert d.shape == (2,)
    g = gsdr_direction(jax.random.PRNGKey(1), theta, loss, jnp.ones(2) * 0.1, alpha=0.25)
    assert g.shape == (2,)
    assert 0.05 <= adapt_alpha(0.2, plateau=True, improving=False) <= 0.8
    assert float(sigmoid_bounded(jnp.array(0.0), Bound(-1.0, 1.0))) == 0.0
    assert float(positive_softplus(jnp.array(0.0))) > 0.0
    manifest = OptimizerManifest("gsdr", 0, 4, 10, alpha=0.5).to_dict()
    assert manifest["truth_mode"] == "truth_safe_unverified"


def test_gsgd_and_pipeline_smoke():
    def loss_fn(x):
        return jnp.sum((x - 1.0) ** 2)

    theta_next, loss = gsgd_step(loss_fn, jnp.array([0.0, 2.0]), 0.1)
    assert float(loss) == 2.0
    assert theta_next.shape == (2,)
    out = run_izhikevich_constant_current(T_ms=50.0, dt_ms=0.5, current_in=10.0)
    assert out["n_spikes"] >= 1
