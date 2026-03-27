"""
systems/networks/omission_two_column.py

Two-column (V1 + Higher-Order/FEF) omission paradigm network.

Architecture:
  V1  : 200 neurons (omission_v1_column.build_v1_column)
  HO  : 100 neurons (70 Pyr, 20 PV, 10 SST)

Inter-areal motifs (Bastos et al. 2015 / Markov laminar logic):
  FF  V1-L2/3 → HO-L4    AMPA          p=0.20, g=0.5
  FB  HO-L5/6 → V1-L2/3  AMPA + NMDA   p=0.10, g=0.3

Four simulation contexts:
  Context 0: BU=ON,  TD=OFF  — pure feedforward stimulus
  Context 1: BU=OFF, TD=OFF  — spontaneous silence
  Context 2: BU=ON,  TD=ON   — attended stimulus
  Context 3: BU=OFF, TD=ON   — OMISSION (prediction without input)
"""

import numpy as np
import jaxley as jx
from jaxley.connect import sparse_connect
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Tuple
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))
from core.mechanisms.models import (
    SafeHH, Inoise,
    GradedAMPA, GradedGABAa, GradedGABAb, GradedNMDA,
)
from systems.networks.omission_v1_column import (
    build_v1_column, V1PopIndices,
    generate_sensory_input, make_stim_schedule,
    _set_pyr_params, _set_pv_params, _set_sst_params,
)


# ── Higher-Order Column ────────────────────────────────────────────────────────

@dataclass
class HOPopIndices:
    l23_pyr:  List[int]
    l4_pyr:   List[int]
    l56_pyr:  List[int]
    pv:       List[int]
    sst:      List[int]

    @property
    def all_pyr(self) -> List[int]:
        return self.l23_pyr + self.l4_pyr + self.l56_pyr

    @property
    def all(self) -> List[int]:
        return self.all_pyr + self.pv + self.sst


def _build_ho_cells(
    n_l23: int, n_l4: int, n_l56: int,
    n_pv: int, n_sst: int,
    offset: int,
    seed: int,
) -> Tuple[List[jx.Cell], HOPopIndices]:
    rng = np.random.default_rng(seed)
    cells: List[jx.Cell] = []
    pops = HOPopIndices(l23_pyr=[], l4_pyr=[], l56_pyr=[], pv=[], sst=[])
    idx = offset

    def _pyr(noise: float) -> jx.Cell:
        soma, dend = jx.Compartment(), jx.Compartment()
        c = jx.Cell([soma, dend], parents=[-1, 0])
        c.radius, c.length = 1.0, 100.0
        c.insert(SafeHH())
        c.insert(Inoise(
            initial_amp_noise=float(np.clip(rng.uniform(noise*0.5, noise*1.5), 0.0, 0.3)),
            initial_tau=20.0,
        ))
        _set_pyr_params(c)
        return c

    def _inh(setter, noise: float) -> jx.Cell:
        c = jx.Cell([jx.Compartment()], parents=[-1])
        c.radius, c.length = 1.0, 10.0
        c.insert(SafeHH())
        c.insert(Inoise(
            initial_amp_noise=float(np.clip(rng.uniform(noise*0.5, noise*1.5), 0.0, 0.3)),
            initial_tau=10.0,
        ))
        setter(c)
        return c

    for i in range(n_l23):
        pops.l23_pyr.append(idx + i); cells.append(_pyr(0.03))
    idx += n_l23

    for i in range(n_l4):
        pops.l4_pyr.append(idx + i); cells.append(_pyr(0.05))
    idx += n_l4

    for i in range(n_l56):
        pops.l56_pyr.append(idx + i); cells.append(_pyr(0.04))
    idx += n_l56

    for i in range(n_pv):
        pops.pv.append(idx + i); cells.append(_inh(_set_pv_params, 0.04))
    idx += n_pv

    for i in range(n_sst):
        pops.sst.append(idx + i); cells.append(_inh(_set_sst_params, 0.02))
    idx += n_sst

    return cells, pops


# ── Trial config ───────────────────────────────────────────────────────────────

@dataclass
class OmissionTrialConfig:
    t_total_ms:    float = 5000.0
    dt_ms:         float = 0.025        # 40 kHz — Jaxley default units
    stim_period_ms:float = 500.0        # stimulus every 500 ms
    omission_ms:   float = 2500.0       # omission starts here
    stim_amp:      float = 2.0
    td_amp:        float = 0.5          # feedback prediction current to V1 L2/3
    pulse_width_ms:float = 20.0

    # context encoding
    bu_on: bool = True
    td_on: bool = False

    @property
    def n_steps(self) -> int:
        return int(self.t_total_ms / self.dt_ms)


# ── Combined network builder ───────────────────────────────────────────────────

@dataclass
class OmissionNetwork:
    net:     jx.Network
    v1_pops: V1PopIndices
    ho_pops: HOPopIndices
    n_v1:    int = 200
    n_ho:    int = 100


def build_omission_network(
    seed: int = 42,
    # V1 sizes (total=200)
    v1_l23: int = 56, v1_l4: int = 40, v1_l56: int = 64,
    v1_pv:  int = 20, v1_sst: int = 12, v1_vip: int = 8,
    # HO sizes (total=100)
    ho_l23: int = 28, ho_l4: int = 20, ho_l56: int = 22,
    ho_pv:  int = 20, ho_sst: int = 10,
    # Inter-areal weights
    g_ff_ampa: float = 0.50,
    g_fb_ampa: float = 0.30, g_fb_nmda: float = 0.20,
    p_ff: float = 0.20, p_fb: float = 0.10,
) -> OmissionNetwork:
    """
    Build and fully wire the two-column omission network (300 neurons).
    """
    np.random.seed(seed)

    # ── V1 column (cells 0..199) ─────────────────────────────────────────────
    v1_net, v1_pops = build_v1_column(
        n_l23=v1_l23, n_l4=v1_l4, n_l56=v1_l56,
        n_pv=v1_pv, n_sst=v1_sst, n_vip=v1_vip, seed=seed,
    )
    v1_cells = list(v1_net.cells)
    n_v1 = len(v1_cells)

    # ── HO column (cells 200..299, offset applied) ───────────────────────────
    ho_cells, ho_pops = _build_ho_cells(
        ho_l23, ho_l4, ho_l56, ho_pv, ho_sst,
        offset=n_v1, seed=seed + 1,
    )
    n_ho = len(ho_cells)

    # ── Combined network ─────────────────────────────────────────────────────
    net = jx.Network(v1_cells + ho_cells)

    # ── Intra-V1 wiring (re-apply into combined net) ─────────────────────────
    # E→E recurrent
    sparse_connect(
        net.cell(v1_pops.all_pyr).branch(0).loc(0.0),
        net.cell(v1_pops.all_pyr).branch(0).loc(0.0),
        GradedAMPA(g=0.5), p=0.08,
    )
    # L4→L23
    sparse_connect(
        net.cell(v1_pops.l4_pyr).branch(0).loc(0.0),
        net.cell(v1_pops.l23_pyr).branch(0).loc(0.0),
        GradedAMPA(g=1.5), p=0.25,
    )
    # L23→L56
    sparse_connect(
        net.cell(v1_pops.l23_pyr).branch(0).loc(0.0),
        net.cell(v1_pops.l56_pyr).branch(0).loc(0.0),
        GradedAMPA(g=0.8), p=0.15,
    )
    # PV→Pyr
    sparse_connect(
        net.cell(v1_pops.pv).branch(0).loc(0.0),
        net.cell(v1_pops.all_pyr).branch(0).loc(0.0),
        GradedGABAa(g=5.0), p=0.5,
    )
    # Pyr→PV
    sparse_connect(
        net.cell(v1_pops.all_pyr).branch(0).loc(0.0),
        net.cell(v1_pops.pv).branch(0).loc(0.0),
        GradedAMPA(g=1.0), p=0.4,
    )
    # SST→Pyr
    sparse_connect(
        net.cell(v1_pops.sst).branch(0).loc(0.0),
        net.cell(v1_pops.all_pyr).branch(0).loc(0.0),
        GradedGABAb(g=1.5), p=0.35,
    )
    # Pyr→SST
    sparse_connect(
        net.cell(v1_pops.all_pyr).branch(0).loc(0.0),
        net.cell(v1_pops.sst).branch(0).loc(0.0),
        GradedAMPA(g=0.8), p=0.3,
    )
    # VIP→SST
    sparse_connect(
        net.cell(v1_pops.vip).branch(0).loc(0.0),
        net.cell(v1_pops.sst).branch(0).loc(0.0),
        GradedGABAa(g=3.0), p=0.6,
    )
    # Pyr→VIP
    sparse_connect(
        net.cell(v1_pops.all_pyr).branch(0).loc(0.0),
        net.cell(v1_pops.vip).branch(0).loc(0.0),
        GradedAMPA(g=0.5), p=0.3,
    )

    # ── Intra-HO wiring ──────────────────────────────────────────────────────
    sparse_connect(
        net.cell(ho_pops.all_pyr).branch(0).loc(0.0),
        net.cell(ho_pops.all_pyr).branch(0).loc(0.0),
        GradedAMPA(g=0.8), p=0.10,
    )
    sparse_connect(
        net.cell(ho_pops.l4_pyr).branch(0).loc(0.0),
        net.cell(ho_pops.l23_pyr).branch(0).loc(0.0),
        GradedAMPA(g=1.2), p=0.25,
    )
    sparse_connect(
        net.cell(ho_pops.pv).branch(0).loc(0.0),
        net.cell(ho_pops.all_pyr).branch(0).loc(0.0),
        GradedGABAa(g=5.0), p=0.5,
    )
    sparse_connect(
        net.cell(ho_pops.all_pyr).branch(0).loc(0.0),
        net.cell(ho_pops.pv).branch(0).loc(0.0),
        GradedAMPA(g=1.0), p=0.4,
    )
    sparse_connect(
        net.cell(ho_pops.sst).branch(0).loc(0.0),
        net.cell(ho_pops.all_pyr).branch(0).loc(0.0),
        GradedGABAb(g=1.5), p=0.35,
    )

    # ── FF: V1-L2/3 → HO-L4 (AMPA, gamma-band driving) ──────────────────────
    sparse_connect(
        net.cell(v1_pops.l23_pyr).branch(0).loc(0.0),
        net.cell(ho_pops.l4_pyr).branch(0).loc(0.0),
        GradedAMPA(g=g_ff_ampa), p=p_ff,
    )

    # ── FB: HO-L5/6 → V1-L2/3 (AMPA + NMDA, alpha/beta feedback) ────────────
    sparse_connect(
        net.cell(ho_pops.l56_pyr).branch(0).loc(0.0),
        net.cell(v1_pops.l23_pyr).branch(0).loc(0.0),
        GradedAMPA(g=g_fb_ampa), p=p_fb,
    )
    sparse_connect(
        net.cell(ho_pops.l56_pyr).branch(0).loc(0.0),
        net.cell(v1_pops.l23_pyr).branch(0).loc(0.0),
        GradedNMDA(g=g_fb_nmda, tauD_NMDA=100.0), p=p_fb,
    )

    print(f"✅ Omission network built: V1={n_v1} HO={n_ho} Total={n_v1+n_ho}")
    return OmissionNetwork(net=net, v1_pops=v1_pops, ho_pops=ho_pops,
                           n_v1=n_v1, n_ho=n_ho)


# ── Context input arrays ───────────────────────────────────────────────────────

def make_context_inputs(
    config: OmissionTrialConfig,
    v1_pops: V1PopIndices,
    ho_pops: HOPopIndices,
    n_v1: int,
) -> Dict[str, np.ndarray]:
    """
    Returns per-cell current injection arrays shape (N_cells, T_steps).
    Context selection: BU=config.bu_on, TD=config.td_on.

    Context 0: BU=ON,  TD=OFF  — feedforward only
    Context 1: BU=OFF, TD=OFF  — silence
    Context 2: BU=ON,  TD=ON   — attended stimulus
    Context 3: BU=OFF, TD=ON   — OMISSION (prediction without input)
    """
    t = np.arange(0, config.t_total_ms, config.dt_ms)
    T = len(t)
    n_total = n_v1 + len(ho_pops.all)
    currents = np.zeros((n_total, T), dtype=np.float64)

    # ── Bottom-up (sensory) input → L4 Pyr ──────────────────────────────────
    if config.bu_on:
        active_stims, _ = make_stim_schedule(
            t_total_ms=config.t_total_ms,
            stim_period_ms=config.stim_period_ms,
            omission_from_ms=None,  # BU=ON means all stimuli delivered
        )
        inp = generate_sensory_input(
            t, active_stims,
            stim_amp=config.stim_amp,
            pulse_width_ms=config.pulse_width_ms,
        )
        for cell_idx in v1_pops.l4_pyr:
            currents[cell_idx] += inp

    # ── Top-down (prediction) input → V1 L2/3 from HO ───────────────────────
    # When TD=ON, HO L5/6 cells receive a sustained rhythmic bias (beta 20 Hz)
    # to model descending prediction signals.
    if config.td_on:
        td_freq_hz   = 20.0                         # beta-band prediction
        td_current   = config.td_amp * np.sin(2 * np.pi * td_freq_hz * t / 1000.0)
        td_current   = np.clip(td_current, 0.0, None)  # half-wave rectify
        for cell_idx in ho_pops.l56_pyr:
            currents[cell_idx] += td_current

    return currents


# ── LFP extractor (mean Vm proxy) ─────────────────────────────────────────────

def extract_lfp(
    traces: np.ndarray,
    pop_indices: List[int],
) -> np.ndarray:
    """Mean somatic Vm of population → synthetic LFP proxy."""
    if not pop_indices:
        return np.zeros(traces.shape[-1])
    return np.mean(traces[pop_indices], axis=0)


# ── Spike detector ────────────────────────────────────────────────────────────

def detect_spikes(
    traces: np.ndarray,
    threshold_mv: float = -20.0,
) -> Dict[int, np.ndarray]:
    """Returns {cell_idx: crossing_timestep_indices} spike dict."""
    spikes = {}
    for i in range(traces.shape[0]):
        v = traces[i]
        crossings = np.where((v[:-1] < threshold_mv) & (v[1:] >= threshold_mv))[0]
        if len(crossings):
            spikes[i] = crossings
    return spikes


# ── Smoke test ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Building omission network (V1=200, HO=100)...")
    onet = build_omission_network(seed=42)
    print(f"Total cells: {onet.n_v1 + onet.n_ho}")
    print(f"HO L4 Pyr (first 5): {onet.ho_pops.l4_pyr[:5]}")

    config = OmissionTrialConfig(bu_on=False, td_on=True)  # omission context
    ctx = make_context_inputs(config, onet.v1_pops, onet.ho_pops, onet.n_v1)
    print(f"Current array shape: {ctx.shape}")
    print("✅ omission_two_column smoke test passed")
