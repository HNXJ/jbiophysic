"""Plotly time-frequency spectrogram with smooth upsampling."""
import numpy as np
import plotly.graph_objects as go
from scipy import signal as sp_signal
from scipy.ndimage import gaussian_filter1d, zoom

def plot_spectrogram(report, window_ms=250.0, overlap=0.95, f_min=1.0, f_max=100.0,
                     title="Time-Frequency Spectrogram", upsample=4) -> go.Figure:
    traces = report.traces; dt = report.dt
    ds = max(1, int(1.0/dt)); ds_traces = traces[:, ::ds]; dt_ds = dt*ds; fs = 1000.0/dt_ds
    threshold = -20.0
    spike_mat = np.zeros_like(ds_traces)
    for i in range(ds_traces.shape[0]):
        crossings = (ds_traces[i, :-1] < threshold) & (ds_traces[i, 1:] >= threshold)
        spike_mat[i, 1:][crossings] = 1.0
    smoothed = gaussian_filter1d(spike_mat.astype(float), sigma=5.0/dt_ds, axis=1)
    mean_signal = np.mean(smoothed, axis=0)
    nperseg = max(1, int(window_ms*fs/1000.0)); noverlap = int(nperseg*overlap)
    frequencies, times, Sxx = sp_signal.spectrogram(mean_signal, fs=fs, window="hann",
                                                     nperseg=nperseg, noverlap=noverlap)
    freq_mask = (frequencies >= f_min) & (frequencies <= f_max)
    frequencies = frequencies[freq_mask]; Sxx = Sxx[freq_mask, :]
    if Sxx.size == 0:
        fig = go.Figure(); fig.add_annotation(text="Insufficient data", x=0.5, y=0.5, showarrow=False)
        return fig
    # Normalize
    Sxx_norm = np.sqrt((Sxx - Sxx.min()) / (Sxx.max() - Sxx.min() + 1e-12))
    # Bicubic upsample for smooth appearance
    if upsample > 1 and Sxx_norm.shape[0] > 2 and Sxx_norm.shape[1] > 2:
        Sxx_norm = zoom(Sxx_norm, (upsample, upsample), order=3)
        frequencies = np.linspace(frequencies[0], frequencies[-1], Sxx_norm.shape[0])
        times = np.linspace(times[0], times[-1], Sxx_norm.shape[1])
    fig = go.Figure()
    fig.add_trace(go.Heatmap(z=Sxx_norm, x=times*1000, y=frequencies, colorscale="Jet",
                              colorbar=dict(title="Power"), zsmooth="best"))
    fig.update_layout(title=dict(text=title, font=dict(size=16, family="Inter")),
        xaxis_title="Time (ms)", yaxis_title="Frequency (Hz)", template="plotly_dark",
        height=400, margin=dict(l=60, r=20, t=50, b=50))
    return fig
