"""
core/optimizers/omission_objective.py

Composite loss and GSDR tuning loop for the omission model.

Objective:
    L = α·RSA + β·(1 − Kappa) + γ·SSS

where:
  RSA   — population-vector cosine distance (↓ = better match to target spikes)
  Kappa — burst synchrony (we *penalise* over-synchrony: want Kappa < 0.1)
  SSS   — spectral similarity (↓ = better match to empirical alpha/beta LFP)

GSDR loop perturbs synaptic params (gAMPA, gGABAa, gGABAb) with Gaussian noise,
keeps improvement, decays step-size by EMA of reward variance.
"""

import numpy as np
import jax
import jax.numpy as jnp
import jaxley as jx
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Callable, Tuple
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from jbiophysics.core.optimizers.omission_metrics import (
    compute_rsa, compute_fleiss_kappa, compute_sss,
    empirical_omission_target_lfp,
)


# ── Composite loss ─────────────────────────────────────────────────────────────

@dataclass
class OmissionLoss:
    alpha: float = 1.0   # RSA weight
    beta:  float = 0.5   # Kappa penalty weight
    gamma: float = 2.0   # SSS weight (spectral is key target)

    def __call__(
        self,
        spikes_sim:   Dict[int, np.ndarray],
        spikes_tgt:   Dict[int, np.ndarray],
        layers:       Dict[str, List[int]],
        lfp_sim:      np.ndarray,
        lfp_tgt:      np.ndarray,
        n_steps:      int,
        fs:           float,
        pop_indices:  List[int],
    ) -> Tuple[float, Dict[str, float]]:
        rsa   = compute_rsa(spikes_sim, spikes_tgt, pop_indices, n_steps)
        kappa = compute_fleiss_kappa(spikes_sim, layers, n_steps)
        sss   = compute_sss(lfp_sim, lfp_tgt, fs)

        loss  = self.alpha * rsa + self.beta * (1.0 - kappa) + self.gamma * sss
        return float(loss), {"rsa": rsa, "kappa": kappa, "sss": sss, "loss": float(loss)}


# ── GSDR tuning state ──────────────────────────────────────────────────────────

@dataclass
class GSRDState:
    epoch:       int   = 0
    best_loss:   float = float("inf")
    best_params: Dict  = field(default_factory=dict)
    history:     List[Dict] = field(default_factory=list)
    sigma:       float = 0.05    # perturbation std
    sigma_min:   float = 0.005  # floor
    ema_var:     float = 0.0     # EMA of reward variance
    ema_alpha:   float = 0.1     # EMA weight


# ── Simple param extractor / applier (JAXley-agnostic) ────────────────────────

def _extract_gparams(net: jx.Network) -> Dict[str, float]:
    """Read shared synaptic conductances from first param group."""
    params = net.get_parameters()
    result: Dict[str, float] = {}
    for group in params:
        for k, v in group.items():
            arr = np.asarray(v).ravel()
            result[k] = float(arr.mean())  # mean of all independent instances
    return result


def _perturb_params(
    params: List[Dict], sigma: float, rng: np.random.Generator
) -> List[Dict]:
    """Return a perturbed copy of the parameters list."""
    perturbed = []
    for group in params:
        new_group: Dict = {}
        for k, v in group.items():
            arr = np.asarray(v)
            noise = rng.normal(0, sigma, size=arr.shape)
            # keep conductances positive
            new_val = np.clip(arr + noise, 1e-4, None)
            new_group[k] = jnp.array(new_val)
        perturbed.append(new_group)
    return perturbed


# ── Main GSDR tuning loop ──────────────────────────────────────────────────────

def run_gsdr_tuning(
    net:          jx.Network,
    simulate_fn:  Callable,         # fn(net, params, config) → (traces, spikes, lfp)
    loss_fn:      OmissionLoss,
    layers:       Dict[str, List[int]],
    pop_indices:  List[int],
    spikes_tgt:   Dict[int, np.ndarray],
    lfp_tgt:      np.ndarray,
    n_steps:      int,
    fs:           float,
    n_epochs:     int = 50,
    sigma_init:   float = 0.05,
    seed:         int = 0,
    status_store: Optional[Dict] = None,  # mutable dict for API /tuning/status
) -> GSRDState:
    """
    Genetic-Stochastic Delta-Rule loop.

    At each epoch:
      1. Perturb current best params with Gaussian noise σ.
      2. Run simulation with perturbed params.
      3. Compute composite loss.
      4. Accept if improved (greedy selection).
      5. Update σ via EMA of reward variance (AGSDR adaptive step).
    """
    rng    = np.random.default_rng(seed)
    state  = GSRDState(sigma=sigma_init)
    params = net.get_parameters()

    # Initial evaluation
    traces_init, spikes_init, lfp_init = simulate_fn(net, params)
    init_loss, init_metrics = loss_fn(
        spikes_init, spikes_tgt, layers, lfp_init, lfp_tgt,
        n_steps, fs, pop_indices,
    )
    state.best_loss   = init_loss
    state.best_params = params
    state.history.append({**init_metrics, "epoch": 0, "sigma": state.sigma})
    print(f"[GSDR] Epoch 0 | loss={init_loss:.4f} | {init_metrics}")

    if status_store is not None:
        status_store.update({"epoch": 0, "loss": init_loss, **init_metrics})

    for epoch in range(1, n_epochs + 1):
        cand_params  = _perturb_params(state.best_params, state.sigma, rng)
        traces, spikes, lfp = simulate_fn(net, cand_params)
        loss, metrics = loss_fn(
            spikes, spikes_tgt, layers, lfp, lfp_tgt,
            n_steps, fs, pop_indices,
        )

        reward = state.best_loss - loss   # positive = improvement

        # EMA variance of rewards → adaptive sigma
        state.ema_var = (1 - state.ema_alpha) * state.ema_var \
                      + state.ema_alpha * reward**2
        reward_std = float(np.sqrt(state.ema_var + 1e-12))
        # Adaptive sigma: grows with high variance, shrinks otherwise
        state.sigma = float(np.clip(
            state.sigma * (1.0 + 0.1 * np.sign(reward_std - 0.01)),
            state.sigma_min, 0.5,
        ))

        if loss < state.best_loss:
            state.best_loss   = loss
            state.best_params = cand_params

        state.epoch = epoch
        record = {**metrics, "epoch": epoch, "sigma": state.sigma}
        state.history.append(record)

        if epoch % 5 == 0 or epoch == 1:
            print(f"[GSDR] Epoch {epoch:3d} | loss={loss:.4f} | "
                  f"rsa={metrics['rsa']:.3f} "
                  f"kappa={metrics['kappa']:.3f} "
                  f"sss={metrics['sss']:.3f} | "
                  f"σ={state.sigma:.4f}")

        if status_store is not None:
            status_store.update({"epoch": epoch, "loss": state.best_loss, **metrics})

    return state
