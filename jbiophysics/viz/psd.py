"""Plotly PSD plots with band annotations."""
import numpy as np
import plotly.graph_objects as go

BAND_DEFINITIONS = {"delta": (1,4), "theta": (4,8), "alpha": (8,13),
                    "beta": (13,30), "gamma": (30,80), "high_gamma": (80,150)}
BAND_COLORS = {"delta": "rgba(156,39,176,0.15)", "theta": "rgba(33,150,243,0.15)",
               "alpha": "rgba(76,175,80,0.15)", "beta": "rgba(255,152,0,0.15)",
               "gamma": "rgba(244,67,54,0.15)", "high_gamma": "rgba(121,85,72,0.15)"}

def compute_psd_numpy(signal_1d, dt, f_max=100.0):
    N = len(signal_1d); fs = 1000.0/dt
    fft_vals = np.fft.rfft(signal_1d); freqs = np.fft.rfftfreq(N, d=dt/1000.0)
    psd = np.abs(fft_vals)**2 / (N*fs); mask = freqs <= f_max
    return freqs[mask], psd[mask]

def plot_psd(report, window=None, f_max=100.0, show_bands=True,
             title="Power Spectral Density") -> go.Figure:
    traces = report.traces; dt = report.dt
    if window:
        traces = traces[:, int(window[0]/dt):int(window[1]/dt)]
    mean_signal = np.mean(traces, axis=0)
    freqs, psd = compute_psd_numpy(mean_signal, dt, f_max)
    psd_norm = psd / (np.max(psd) + 1e-12)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=freqs, y=psd_norm, mode="lines",
        line=dict(color="#4FC3F7", width=2), fill="tozeroy",
        fillcolor="rgba(79,195,247,0.15)", name="PSD"))
    if show_bands:
        for band, (f_lo, f_hi) in BAND_DEFINITIONS.items():
            if f_hi <= f_max:
                fig.add_vrect(x0=f_lo, x1=f_hi, fillcolor=BAND_COLORS.get(band),
                    layer="below", line_width=0, annotation_text=band,
                    annotation_position="top left", annotation_font_size=9,
                    annotation_font_color="white")
    fig.update_layout(title=dict(text=title, font=dict(size=16, family="Inter")),
        xaxis_title="Frequency (Hz)", yaxis_title="Normalized Power",
        template="plotly_dark", height=350, margin=dict(l=60, r=20, t=50, b=50))
    return fig
