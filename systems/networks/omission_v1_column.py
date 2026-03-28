"""
systems/networks/omission_v1_column.py

200-neuron V1 laminar column for the omission paradigm.

Cell-type ratios (Markram et al. / Bastos laminar conventions):
  L2/3 Pyr : 56    L4 Pyr  : 40    L5/6 Pyr : 64
  PV        : 20   SST     : 12    VIP      :  8   (total: 200)

Feedforward sensory input driven into L4 Pyr population.
Returns (net, meta, pop_indices) for downstream wiring.
"""

import numpy as np
import jax.numpy as jnp
import jaxley as jx
from jaxley.connect import sparse_connect
from typing import Optional, Dict, Tuple, List
from dataclasses import dataclass

# ── local imports ──────────────────────────────────────────────────────────────
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..")) # Point to project root
from core.mechanisms.models import (
    SafeHH, Inoise,
    GradedAMPA, GradedGABAa, GradedGABAb, GradedNMDA,
)


# ── Cell-type-specific HH overrides ───────────────────────────────────────────

def _set_pyr_params(cell: jx.Cell) -> jx.Cell:
    """Regular-spiking pyramidal: standard HH conductances."""
    cell.set("HH_gNa", 120.0)
    cell.set("HH_gK",   36.0)
    cell.set("HH_gLeak", 0.3)
    return cell

def _set_pv_params(cell: jx.Cell) -> jx.Cell:
    """Fast-spiking PV: high Na/K for narrow spikes, fast repolarization."""
    cell.set("HH_gNa", 200.0)
    cell.set("HH_gK",   72.0)
    cell.set("HH_gLeak", 0.5)
    return cell

def _set_sst_params(cell: jx.Cell) -> jx.Cell:
    """Low-threshold SST: intermediate conductances, dendritic targeting."""
    cell.set("HH_gNa", 100.0)
    cell.set("HH_gK",   28.0)
    cell.set("HH_gLeak", 0.3)
    return cell

def _set_vip_params(cell: jx.Cell) -> jx.Cell:
    """Disinhibitory VIP: small cell, lower conductances."""
    cell.set("HH_gNa",  80.0)
    cell.set("HH_gK",   20.0)
    cell.set("HH_gLeak", 0.2)
    return cell


# ── Builder ────────────────────────────────────────────────────────────────────

@dataclass
class V1PopIndices:
    l23_pyr:  List[int]
    l4_pyr:   List[int]
    l56_pyr:  List[int]
    pv:       List[int]
    sst:      List[int]
    vip:      List[int]

    @property
    def all_pyr(self) -> List[int]:
        return self.l23_pyr + self.l4_pyr + self.l56_pyr

    @property
    def all_inh(self) -> List[int]:
        return self.pv + self.sst + self.vip

    @property
    def all(self) -> List[int]:
        return self.all_pyr + self.all_inh


def create_v1_cells(
    n_l23: int = 56,
    n_l4:  int = 40,
    n_l56: int = 64,
    n_pv:  int = 20,
    n_sst: int = 12,
    n_vip: int = 8,
    seed:  Optional[int] = 0,
) -> Tuple[List[jx.Cell], V1PopIndices]:
    """
    Create a list of 200 neurons with Lamina/Cell-type identities.

    Returns:
        cells – List[jx.Cell]
        pops  – V1PopIndices dataclass with relative cell indices
    """
    np.random.seed(seed)
    total = n_l23 + n_l4 + n_l56 + n_pv + n_sst + n_vip

    all_cells: List[jx.Cell] = []
    pops = V1PopIndices(
        l23_pyr=[], l4_pyr=[], l56_pyr=[], pv=[], sst=[], vip=[]
    )
    offset = 0

    def _make_pyr(noise_amp: float) -> jx.Cell:
        soma = jx.Compartment()
        dend = jx.Compartment()
        cell = jx.Cell([soma, dend], parents=[-1, 0])
        cell.radius = 1.0
        cell.length = 100.0
        cell.insert(SafeHH(name="HH"))
        cell.insert(Inoise(
            initial_amp_noise=float(np.clip(np.random.uniform(noise_amp*0.5, noise_amp*1.5), 0.0, 0.3)),
            initial_tau=20.0,
        ))
        _set_pyr_params(cell)
        return cell

    def _make_inh(setter_fn, noise_amp: float) -> jx.Cell:
        soma = jx.Compartment()
        cell = jx.Cell([soma], parents=[-1])
        cell.radius = 1.0
        cell.length = 10.0
        cell.insert(SafeHH(name="HH"))
        cell.insert(Inoise(
            initial_amp_noise=float(np.clip(np.random.uniform(noise_amp*0.5, noise_amp*1.5), 0.0, 0.3)),
            initial_tau=10.0,
        ))
        setter_fn(cell)
        return cell

    # L2/3 Pyr
    for i in range(n_l23):
        pops.l23_pyr.append(offset + i)
        all_cells.append(_make_pyr(0.03))
    offset += n_l23

    # L4 Pyr
    for i in range(n_l4):
        pops.l4_pyr.append(offset + i)
        all_cells.append(_make_pyr(0.05))
    offset += n_l4

    # L5/6 Pyr
    for i in range(n_l56):
        pops.l56_pyr.append(offset + i)
        all_cells.append(_make_pyr(0.04))
    offset += n_l56

    # PV
    for i in range(n_pv):
        pops.pv.append(offset + i)
        all_cells.append(_make_inh(_set_pv_params, 0.04))
    offset += n_pv

    # SST
    for i in range(n_sst):
        pops.sst.append(offset + i)
        all_cells.append(_make_inh(_set_sst_params, 0.02))
    offset += n_sst

    # VIP
    for i in range(n_vip):
        pops.vip.append(offset + i)
        all_cells.append(_make_inh(_set_vip_params, 0.02))
    offset += n_vip

    assert offset == total
    return all_cells, pops

def wire_v1_column(net: jx.Network, pops: V1PopIndices):
    """
    Apply intra-column synaptic wiring to a jaxley Network.
    
    Args:
        net  - jaxley Network (already containing V1 cells)
        pops - V1PopIndices (relative or absolute, must match net indexing)
    """
    # ── Intra-column synaptic wiring ──────────────────────────────────────────
    # E→all  (AMPA, recurrent excitation)
    sparse_connect(
        net.cell(pops.all_pyr).branch(0).loc(0.0),
        net.cell(pops.all_pyr).branch(0).loc(0.0),
        GradedAMPA(g=0.5), p=0.08,
    )

    # L4→L23 feedforward drive (strong)
    sparse_connect(
        net.cell(pops.l4_pyr).branch(0).loc(0.0),
        net.cell(pops.l23_pyr).branch(0).loc(0.0),
        GradedAMPA(g=1.5), p=0.25,
    )

    # L23→L56 downward drive
    sparse_connect(
        net.cell(pops.l23_pyr).branch(0).loc(0.0),
        net.cell(pops.l56_pyr).branch(0).loc(0.0),
        GradedAMPA(g=0.8), p=0.15,
    )

    # PV→Pyr  (fast perisomatic inhibition, GABAa)
    sparse_connect(
        net.cell(pops.pv).branch(0).loc(0.0),
        net.cell(pops.all_pyr).branch(0).loc(0.0),
        GradedGABAa(g=5.0), p=0.5,
    )

    # Pyr→PV  (recruit local interneurons)
    sparse_connect(
        net.cell(pops.all_pyr).branch(0).loc(0.0),
        net.cell(pops.pv).branch(0).loc(0.0),
        GradedAMPA(g=1.0), p=0.4,
    )

    # SST→Pyr  (slow dendritic inhibition, GABAb)
    sparse_connect(
        net.cell(pops.sst).branch(0).loc(0.0),
        net.cell(pops.all_pyr).branch(0).loc(0.0),
        GradedGABAb(g=1.5), p=0.35,
    )

    # Pyr→SST
    sparse_connect(
        net.cell(pops.all_pyr).branch(0).loc(0.0),
        net.cell(pops.sst).branch(0).loc(0.0),
        GradedAMPA(g=0.8), p=0.3,
    )

    # VIP→SST  (disinhibition, GABAa)
    sparse_connect(
        net.cell(pops.vip).branch(0).loc(0.0),
        net.cell(pops.sst).branch(0).loc(0.0),
        GradedGABAa(g=3.0), p=0.6,
    )

    # Pyr→VIP  (recruit disinhibitory circuit)
    sparse_connect(
        net.cell(pops.all_pyr).branch(0).loc(0.0),
        net.cell(pops.vip).branch(0).loc(0.0),
        GradedAMPA(g=0.5), p=0.3,
    )

    print(f"✅ V1 column wired into network.")

def build_v1_column(*args, **kwargs) -> Tuple[jx.Network, V1PopIndices]:
    """Standalone builder: creates cells, creates network, and wires it."""
    cells, pops = create_v1_cells(*args, **kwargs)
    net = jx.Network(cells)
    wire_v1_column(net, pops)
    return net, pops


# ── Sensory input generator ────────────────────────────────────────────────────

def generate_sensory_input(
    t_ms: np.ndarray,
    stim_times_ms: np.ndarray,
    stim_amp: float = 2.0,
    pulse_width_ms: float = 20.0,
    jitter_ms: float = 2.0,
    seed: int = 1,
) -> np.ndarray:
    """
    Generate a current array to inject into L4 Pyr cells.

    Args:
        t_ms           – time vector [ms], shape (T,)
        stim_times_ms  – onset times of each expected stimulus [ms]
        stim_amp       – peak current amplitude [µA/cm²]
        pulse_width_ms – duration of each pulse [ms]
        jitter_ms      – Gaussian jitter on stimulus onset [ms]

    Returns:
        current [µA/cm²], shape (T,)  — inject into l4_pyr as current_clamp
    """
    rng = np.random.default_rng(seed)
    current = np.zeros_like(t_ms, dtype=np.float64)
    for t_on in stim_times_ms:
        t_jit = t_on + rng.normal(0, jitter_ms)
        mask = (t_ms >= t_jit) & (t_ms < t_jit + pulse_width_ms)
        # Smooth Hanning envelope
        n_pts = np.sum(mask)
        if n_pts > 0:
            envelope = np.hanning(n_pts) * stim_amp
            current[mask] += envelope
    return current


def make_stim_schedule(
    t_total_ms: float = 5000.0,
    stim_period_ms: float = 500.0,
    omission_from_ms: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns (all_stim_times, omitted_stim_times).

    If omission_from_ms is set, stimuli after that point are omitted (BU=OFF).
    """
    stim_times = np.arange(stim_period_ms, t_total_ms, stim_period_ms)
    if omission_from_ms is not None:
        active = stim_times[stim_times < omission_from_ms]
        omitted = stim_times[stim_times >= omission_from_ms]
        return active, omitted
    return stim_times, np.array([])


if __name__ == "__main__":
    print("Building V1 column (200 neurons)...")
    net, pops = build_v1_column(seed=42)
    print(f"Total cells: {len(pops.all)}")
    print(f"L4 Pyr indices (first 5): {pops.l4_pyr[:5]}")

    # Quick sensory input test
    t = np.arange(0, 100, 0.025)
    stim_times = np.array([20.0, 60.0])
    inp = generate_sensory_input(t, stim_times, stim_amp=2.0)
    print(f"Input peak: {inp.max():.3f}  shape: {inp.shape}")
    print("✅ omission_v1_column smoke test passed")
