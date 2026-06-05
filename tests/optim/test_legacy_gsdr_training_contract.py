"""
Legacy GSDR/AGSDR training contract compatibility tests.

Tests the API contract from the legacy training notebook:
- loss_fn(params, inputs, labels) returns (scalar_loss, traces)
- jax.value_and_grad(loss_fn, has_aux=True) workflow
- MCDP-like factors computed from traces
- optimizer.update(..., params=..., value=..., key=..., mcdp_factors=...)
- optax.apply_updates

No biological validation claims. API compatibility only.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import optax
import pytest

from jbiophysic.optim.agsdr import AGSDR
from jbiophysic.optim.gsdr import GSDR
from jbiophysic.optim.gsgd import GSGD
from jbiophysic.optim.sdr import SDR


def toy_loss(params, x, y):
    """Toy differentiable regression loss with auxiliary traces."""
    pred = x @ params["w"] + params["b"]
    err = pred - y
    loss = jnp.mean(err**2)
    traces = {
        "prediction": pred,
        "error": err,
        "activity": jnp.tanh(pred),
    }
    return loss, traces


def toy_mcdp_factors(params, traces):
    """Toy MCDP-like factors derived from traces (same PyTree as params)."""
    activity_scale = jnp.clip(jnp.mean(jnp.abs(traces["activity"])), 0.1, 2.0)
    return jax.tree.map(lambda p: jnp.ones_like(p) * activity_scale, params)


def assert_tree_finite(tree, name="tree"):
    """Assert all leaves of a PyTree are finite."""
    leaves = jax.tree_util.tree_leaves(tree)
    assert leaves, f"{name} is empty"
    for leaf in leaves:
        assert jnp.all(jnp.isfinite(leaf)), f"{name} contains non-finite values"


@pytest.fixture
def toy_problem():
    """Tiny toy regression problem."""
    params = {
        "w": jnp.array([[0.1], [-0.2]], dtype=jnp.float32),
        "b": jnp.array([0.05], dtype=jnp.float32),
    }
    x = jnp.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], dtype=jnp.float32)
    y = jnp.array([[0.5], [-0.25], [0.25]], dtype=jnp.float32)
    return params, x, y


def test_gsdr_supports_legacy_value_grad_has_aux_contract(toy_problem):
    """Test GSDR with legacy value_and_grad(has_aux=True) contract."""
    params, x, y = toy_problem

    # Initialize optimizer with SDR as inner
    inner_tx = SDR()
    tx = GSDR(inner_optimizer=inner_tx)
    opt_state = tx.init(params)

    # Legacy contract: value_and_grad with has_aux=True
    (loss_val, traces), grads = jax.value_and_grad(toy_loss, has_aux=True)(params, x, y)

    assert_tree_finite(grads, "grads")
    assert jnp.isfinite(loss_val), "loss not finite"

    # Compute MCDP factors from traces
    mcdp_factors = toy_mcdp_factors(params, traces)

    # Legacy update call
    key = jax.random.PRNGKey(0)
    updates, opt_state = tx.update(
        grads,
        opt_state,
        params=params,
        value=loss_val,
        key=key,
        mcdp_factors=mcdp_factors,
    )

    # Verify updates structure
    assert_tree_finite(updates, "updates")

    # Legacy apply_updates
    new_params = optax.apply_updates(params, updates)
    assert_tree_finite(new_params, "new_params")


def test_gsdr_legacy_contract_jitted_step(toy_problem):
    """Test GSDR legacy contract inside jax.jit."""
    params, x, y = toy_problem

    inner_tx = SDR()
    tx = GSDR(inner_optimizer=inner_tx)
    opt_state = tx.init(params)

    def jitted_step(params, opt_state, x, y, key):
        (loss_val, traces), grads = jax.value_and_grad(toy_loss, has_aux=True)(params, x, y)
        mcdp_factors = toy_mcdp_factors(params, traces)
        updates, new_opt_state = tx.update(
            grads,
            opt_state,
            params=params,
            value=loss_val,
            key=key,
            mcdp_factors=mcdp_factors,
        )
        new_params = optax.apply_updates(params, updates)
        return new_params, new_opt_state, loss_val

    jitted_step = jax.jit(jitted_step)

    key = jax.random.PRNGKey(0)
    new_params, new_opt_state, loss_val = jitted_step(params, opt_state, x, y, key)

    assert_tree_finite(new_params, "jitted new_params")
    assert jnp.isfinite(loss_val), "jitted loss not finite"


def test_agsdr_supports_legacy_value_grad_has_aux_contract(toy_problem):
    """Test AGSDR with legacy value_and_grad(has_aux=True) contract."""
    params, x, y = toy_problem

    # Initialize AGSDR with SDR as inner optimizer
    inner_tx = SDR()
    tx = AGSDR(inner_optimizer=inner_tx)
    opt_state = tx.init(params)

    # Legacy contract: value_and_grad with has_aux=True
    (loss_val, traces), grads = jax.value_and_grad(toy_loss, has_aux=True)(params, x, y)

    assert_tree_finite(grads, "grads")
    assert jnp.isfinite(loss_val), "loss not finite"

    # Compute MCDP factors from traces
    mcdp_factors = toy_mcdp_factors(params, traces)

    # Legacy update call (AGSDR supports same signature)
    key = jax.random.PRNGKey(0)
    updates, opt_state = tx.update(
        grads,
        opt_state,
        params=params,
        value=loss_val,
        key=key,
        mcdp_factors=mcdp_factors,
    )

    # Verify updates structure
    assert_tree_finite(updates, "updates")

    # Legacy apply_updates
    new_params = optax.apply_updates(params, updates)
    assert_tree_finite(new_params, "new_params")


def test_gsgd_apply_updates_compatibility(toy_problem):
    """Test GSGD with standard Optax update and apply_updates.

    GSGD is a standard GradientTransformation (not GradientTransformationExtraArgs),
    so it does not support the key parameter. This test verifies basic compatibility
    with the Optax apply_updates workflow.
    """
    params, x, y = toy_problem

    # GSGD is a standard Optax-style optimizer
    tx = GSGD()
    opt_state = tx.init(params)

    # Standard Optax contract (GSGD does not require key)
    (loss_val, traces), grads = jax.value_and_grad(toy_loss, has_aux=True)(params, x, y)

    assert_tree_finite(grads, "grads")
    assert jnp.isfinite(loss_val), "loss not finite"

    # GSGD standard update (no key parameter)
    updates, opt_state = tx.update(grads, opt_state, params=params)

    # Verify updates structure
    assert_tree_finite(updates, "updates")

    # Standard apply_updates
    new_params = optax.apply_updates(params, updates)
    assert_tree_finite(new_params, "new_params")
