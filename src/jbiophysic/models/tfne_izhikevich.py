"""TFNE-Izhikevich calibration bridge.

The bridge is deliberately explicit: Izhikevich `I` is phenomenological and becomes amperes
only through a user-declared scale.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp


@dataclass(frozen=True)
class IzhikevichTFNEScale:
    izh_current_to_ampere_scale: float = 1e-12

    def __post_init__(self) -> None:
        if self.izh_current_to_ampere_scale <= 0:
            raise ValueError("scale must be positive and explicitly declared")


def izh_current_to_ampere(current_native: jnp.ndarray, scale: IzhikevichTFNEScale) -> jnp.ndarray:
    """Convert native Izhikevich current-like drive into amperes by declared calibration."""
    return current_native * scale.izh_current_to_ampere_scale
