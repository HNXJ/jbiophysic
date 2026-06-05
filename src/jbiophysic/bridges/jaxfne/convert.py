"""Convert jbiophysic configs to jaxfne manifests.

Bridge adapters from jbiophysic parameter/config format to jaxfne-compatible dicts.
Does not simulate or execute; only converts structure.
"""

from __future__ import annotations

from typing import Any

from .reports import json_safe


def jbiophysic_params_to_jaxfne(
    cell_type: str,
    jb_params: dict[str, Any],
) -> dict[str, Any]:
    """Convert jbiophysic cell params to jaxfne source config.

    Parameters
    ----------
    cell_type : str
        One of "izhikevich" or "hodgkin_huxley".
    jb_params : Dict[str, Any]
        jbiophysic parameter dict.

    Returns
    -------
    Dict[str, Any]
        jaxfne-compatible source config.

    Raises
    ------
    ValueError
        If cell_type not recognized.
    """
    if cell_type == "izhikevich":
        # Izhikevich: a, b, c, d, I_inj_nA, optional v0, u0
        return {
            "a": float(jb_params.get("a", 0.02)),
            "b": float(jb_params.get("b", 0.2)),
            "c": float(jb_params.get("c", -65.0)),
            "d": float(jb_params.get("d", 8.0)),
            "I_ext_nA": float(jb_params.get("I_inj_nA", 10.0)),
            "v0": float(jb_params.get("v0", -65.0)),
            "u0": float(jb_params.get("u0", 0.0)),
            "extra_params": {
                k: v
                for k, v in jb_params.items()
                if k not in ("a", "b", "c", "d", "I_inj_nA", "v0", "u0")
            },
        }
    elif cell_type == "hodgkin_huxley":
        # Hodgkin-Huxley: g_Na, g_K, g_L, E_Na, E_K, E_L, C_m, I_inj_pA, optional V0
        return {
            "g_Na": float(jb_params.get("g_Na", 120.0)),
            "g_K": float(jb_params.get("g_K", 36.0)),
            "g_L": float(jb_params.get("g_L", 0.3)),
            "E_Na": float(jb_params.get("E_Na", 50.0)),
            "E_K": float(jb_params.get("E_K", -77.0)),
            "E_L": float(jb_params.get("E_L", -54.4)),
            "C_m": float(jb_params.get("C_m", 1.0)),
            "I_ext_pA": float(jb_params.get("I_inj_pA", 10.0)),
            "V0": float(jb_params.get("V0", -65.0)),
            "extra_params": {
                k: v
                for k, v in jb_params.items()
                if k not in ("g_Na", "g_K", "g_L", "E_Na", "E_K", "E_L", "C_m", "I_inj_pA", "V0")
            },
        }
    else:
        raise ValueError(f"Unknown cell_type: {cell_type}")


def jbiophysic_circuit_to_jaxfne(
    n_exc: int,
    n_inh: int,
    jb_connectivity: dict[str, Any],
) -> dict[str, Any]:
    """Convert jbiophysic E/I circuit to jaxfne network manifest.

    Parameters
    ----------
    n_exc : int
        Number of excitatory neurons.
    n_inh : int
        Number of inhibitory neurons.
    jb_connectivity : Dict[str, Any]
        jbiophysic connectivity dict with adjacency, weights, synapse_model.

    Returns
    -------
    Dict[str, Any]
        jaxfne-compatible network manifest.

    Raises
    ------
    ValueError
        If n_exc < 0, n_inh < 0, or total neurons == 0.
    """
    if n_exc < 0:
        raise ValueError(f"n_exc must be >= 0, got {n_exc}")
    if n_inh < 0:
        raise ValueError(f"n_inh must be >= 0, got {n_inh}")
    if n_exc + n_inh == 0:
        raise ValueError("total neurons (n_exc + n_inh) must be > 0")

    adjacency = jb_connectivity.get("adjacency")
    weights = jb_connectivity.get("weights")
    synapse_model = jb_connectivity.get("synapse_model", "exponential")

    return {
        "n_neurons": n_exc + n_inh,
        "n_exc": n_exc,
        "n_inh": n_inh,
        "connectivity_matrix": json_safe(adjacency),
        "synaptic_weights": json_safe(weights),
        "synapse_model": synapse_model,
    }
