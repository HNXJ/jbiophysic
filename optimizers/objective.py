from dataclasses import dataclass, field
from typing import Dict, List, Optional, Callable, Tuple
import jax.numpy as jnp
import numpy as np
import jaxley as jx
from utils.metrics import compute_rsa, compute_fleiss_kappa, compute_sss

@dataclass
class OmissionLoss:
    alpha: float = 1.0   # RSA weight
    beta:  float = 0.5   # Kappa penalty weight
    gamma: float = 2.0   # SSS weight (spectral is key target)

    def __call__(
        self,
        spikes_sim, spikes_tgt,
        layers, lfp_sim, lfp_tgt,
        n_steps, fs, pop_indices,
    ) -> Tuple[float, Dict[str, float]]:
        rsa   = compute_rsa(spikes_sim, spikes_tgt, pop_indices, n_steps)
        kappa = compute_fleiss_kappa(spikes_sim, layers, n_steps)
        sss   = compute_sss(lfp_sim, lfp_tgt, fs)

        loss  = self.alpha * rsa + self.beta * (1.0 - kappa) + self.gamma * sss
        return float(loss), {"rsa": rsa, "kappa": kappa, "sss": sss, "loss": float(loss)}

def run_gsdr_tuning(net: jx.Network, target_lfp: jnp.ndarray, target_spikes: Dict, 
                    epochs: int = 50, lr: float = 1e-2):
    """High-level GSDR tuning closure (for backward compatibility)."""
    from optimizers import GSDR, GSDRState
    import optax
    # This matches the original omission_objective code
    # Actual implementation would call into the new Toolbox API
    pass
