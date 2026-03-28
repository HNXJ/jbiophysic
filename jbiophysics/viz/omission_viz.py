"""
jbiophysics/viz/omission_viz.py

Matplotlib-based visualization for the omission paradigm.
Returns Base64-encoded PNG strings for API embedding.
"""

import io
import base64
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.signal import welch
from scipy.ndimage import gaussian_filter1d
from typing import Dict, List, Optional, Tuple


# ── Theme ──────────────────────────────────────────────────────────────────────
GOLD   = "#CFB87C"
VIOLET = "#9400D3"
CYAN   = "#4FC3F7"
WHITE  = "#E8E8E8"
BG     = "#0D0D0F"

LAYER_COLORS = {
    "l23":  "#00FFFF",
    "l4":   "#CFB87C",
    "l56":  "#9400D3",
    "pv":   "#FF5252",
    "sst":  "#FF9800",
    "vip":  "#4CAF50",
    "ho":   "#7E57C2",
}


def _fig_to_b64(fig: plt.Figure) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight",
                facecolor=BG, edgecolor="none")
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode("utf-8")
    plt.close(fig)
    return b64


# ── Raster plot ────────────────────────────────────────────────────────────────

def plot_omission_raster(
    traces:   np.ndarray,           # (N, T)
    dt_ms:    float,
    pops:     Dict[str, List[int]], # {"l23_pyr": [...], "l4_pyr": [...], ...}
    threshold_mv: float = -20.0,
    omission_onset_ms: Optional[float] = None,
    title: str = "Omission Raster",
) -> str:
    """Returns Base64 PNG of raster coloured by population."""
    n_steps = traces.shape[1]
    t_ms = np.arange(n_steps) * dt_ms

    fig, ax = plt.subplots(figsize=(12, 5), facecolor=BG)
    ax.set_facecolor(BG)

    for pop_name, indices in pops.items():
        color = LAYER_COLORS.get(pop_name.split("_")[0], WHITE)
        for cell_idx in indices:
            if cell_idx >= traces.shape[0]:
                continue
            v = traces[cell_idx]
            crosses = np.where((v[:-1] < threshold_mv) & (v[1:] >= threshold_mv))[0]
            ax.scatter(t_ms[crosses], np.full_like(crosses, cell_idx, dtype=float),
                       c=color, s=1.5, alpha=0.8, rasterized=True)

    if omission_onset_ms is not None:
        ax.axvline(omission_onset_ms, color=GOLD, ls="--", lw=1.5,
                   label=f"Omission @{omission_onset_ms:.0f}ms", alpha=0.9)
        ax.legend(loc="upper right", fontsize=8, facecolor=BG, labelcolor=WHITE)

    ax.set_xlabel("Time (ms)", color=WHITE, fontsize=10)
    ax.set_ylabel("Neuron index", color=WHITE, fontsize=10)
    ax.set_title(title, color=GOLD, fontsize=12, pad=6)
    ax.tick_params(colors=WHITE)
    for spine in ax.spines.values():
        spine.set_edgecolor("#333333")

    return _fig_to_b64(fig)


# ── LFP dual-column traces ─────────────────────────────────────────────────────

def plot_lfp_traces(
    lfp_v1: np.ndarray,
    lfp_ho: np.ndarray,
    dt_ms:  float,
    omission_onset_ms: Optional[float] = None,
    title: str = "LFP Traces — V1 & HO",
) -> str:
    """Returns Base64 PNG of dual-column LFP."""
    t_ms = np.arange(len(lfp_v1)) * dt_ms

    fig, axes = plt.subplots(2, 1, figsize=(12, 4), facecolor=BG, sharex=True)

    for ax, lfp, color, label in zip(
        axes,
        [lfp_v1, lfp_ho],
        [CYAN, VIOLET],
        ["V1 (Primary)", "HO / FEF (Higher-Order)"],
    ):
        ax.set_facecolor(BG)
        # Light smoothing for display
        lfp_sm = gaussian_filter1d(lfp, sigma=4)
        ax.plot(t_ms, lfp_sm, color=color, lw=0.8, alpha=0.9)
        ax.fill_between(t_ms, lfp_sm, alpha=0.12, color=color)
        ax.set_ylabel(label, color=WHITE, fontsize=9)
        ax.tick_params(colors=WHITE)
        for spine in ax.spines.values():
            spine.set_edgecolor("#333333")
        if omission_onset_ms is not None:
            ax.axvline(omission_onset_ms, color=GOLD, ls="--", lw=1.0, alpha=0.8)

    axes[-1].set_xlabel("Time (ms)", color=WHITE, fontsize=10)
    axes[0].set_title(title, color=GOLD, fontsize=12, pad=6)
    fig.tight_layout(pad=0.5)
    return _fig_to_b64(fig)


# ── Time-Frequency Representation (Morlet wavelet) ────────────────────────────

def plot_tfr(
    lfp:   np.ndarray,
    dt_ms: float,
    f_min: float = 4.0,
    f_max: float = 100.0,
    n_freqs: int = 64,
    title: str = "TFR — Omission LFP",
) -> str:
    """
    Returns Base64 PNG of Morlet wavelet TFR.
    Highlights alpha (8-13 Hz) and beta (13-30 Hz) bands.
    """
    # Manual Morlet and CWT to avoid missing scipy.signal.morlet2/cwt in this env
    def _morlet2(n, s, w=5.0):
        return np.exp(-0.5 * (n / s)**2) * np.exp(1j * w * n / s)

    fs     = 1000.0 / dt_ms
    t_ms   = np.arange(len(lfp)) * dt_ms
    freqs  = np.logspace(np.log10(f_min), np.log10(f_max), n_freqs)
    
    # Continuous wavelet transform via convolution
    cwtm = np.zeros((len(freqs), len(lfp)), dtype=complex)
    lfp_norm = lfp - np.mean(lfp)
    
    for i, f in enumerate(freqs):
        s = 5.0 * fs / (2 * np.pi * f) # w=5
        # Convolution kernel width
        M = int(10 * s)
        if M % 2 == 0: M += 1
        n = np.arange(M) - (M - 1) / 2
        kernel = _morlet2(n, s, w=5.0)
        # Normalization to match scipy.signal.cwt roughly
        kernel *= np.power(np.pi, -0.25) * np.sqrt(1/s)
        res = np.convolve(lfp_norm, kernel, mode='same')
        # Ensure 'same' length as lfp_norm (numpy 'same' returns max(len(lfp_norm), len(kernel)))
        if len(res) > len(lfp_norm):
            start = (len(res) - len(lfp_norm)) // 2
            res = res[start : start + len(lfp_norm)]
        cwtm[i, :] = res
    power = np.abs(cwtm) ** 2

    # Normalise per-frequency (dB relative to mean)
    mean_p = power.mean(axis=1, keepdims=True) + 1e-30
    power_db = 10 * np.log10(power / mean_p)

    fig, ax = plt.subplots(figsize=(12, 4), facecolor=BG)
    ax.set_facecolor(BG)

    im = ax.pcolormesh(
        t_ms, freqs, power_db,
        cmap="inferno", shading="auto",
        vmin=-6, vmax=6,
    )

    # Band overlays
    ax.axhspan(8,  13, alpha=0.08, color=GOLD,   label="Alpha (8-13 Hz)")
    ax.axhspan(13, 30, alpha=0.08, color=CYAN,   label="Beta (13-30 Hz)")
    ax.axhspan(30, 80, alpha=0.05, color=VIOLET, label="Gamma (30-80 Hz)")

    ax.set_yscale("log")
    ax.set_ylabel("Frequency (Hz)", color=WHITE, fontsize=10)
    ax.set_xlabel("Time (ms)", color=WHITE, fontsize=10)
    ax.set_title(title, color=GOLD, fontsize=12, pad=6)
    ax.tick_params(colors=WHITE)
    ax.legend(loc="upper right", fontsize=8, facecolor=BG, labelcolor=WHITE)
    for spine in ax.spines.values():
        spine.set_edgecolor("#333333")

    cbar = fig.colorbar(im, ax=ax, pad=0.01)
    cbar.ax.tick_params(colors=WHITE)
    cbar.set_label("Power (dB)", color=WHITE, fontsize=9)

    fig.tight_layout(pad=0.5)
    return _fig_to_b64(fig)
