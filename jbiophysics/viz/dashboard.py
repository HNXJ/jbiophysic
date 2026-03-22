"""Combined multi-panel Plotly dashboard."""
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def generate_dashboard(report, title="Jbiophysics — Simulation Dashboard") -> go.Figure:
    has_loss = len(report.loss_history) > 0
    n_rows = 4 if has_loss else 3
    subtitles = ["Spike Raster", "Time-Frequency", "Power Spectral Density"]
    if has_loss: subtitles.append("Optimization Loss")
    fig = make_subplots(rows=n_rows, cols=1, subplot_titles=subtitles, vertical_spacing=0.06,
        row_heights=[0.3, 0.3, 0.25] + ([0.15] if has_loss else []))
    traces_data = report.traces; dt = report.dt
    num_cells, num_steps = traces_data.shape; time_ms = np.arange(num_steps) * dt
    threshold = -20.0
    # Panel 1: Raster
    spike_times, spike_neurons = [], []
    for i in range(num_cells):
        crossings = (traces_data[i, :-1] < threshold) & (traces_data[i, 1:] >= threshold)
        indices = np.where(crossings)[0]
        spike_times.extend(time_ms[indices]); spike_neurons.extend([i]*len(indices))
    fig.add_trace(go.Scattergl(x=spike_times, y=spike_neurons, mode="markers",
        marker=dict(size=2, color="#4FC3F7", opacity=0.8), showlegend=False), row=1, col=1)
    fig.update_yaxes(title_text="Neuron", row=1, col=1)
    # Panel 2: Spectrogram
    try:
        from scipy import signal as sp_signal
        from scipy.ndimage import gaussian_filter1d, zoom
        ds = max(1, int(1.0/dt)); ds_traces = traces_data[:, ::ds]; dt_ds = dt*ds; fs = 1000.0/dt_ds
        spike_mat = np.zeros_like(ds_traces)
        for i in range(ds_traces.shape[0]):
            c = (ds_traces[i, :-1] < threshold) & (ds_traces[i, 1:] >= threshold)
            spike_mat[i, 1:][c] = 1.0
        smoothed = gaussian_filter1d(spike_mat.astype(float), sigma=5.0/dt_ds, axis=1)
        mean_sig = np.mean(smoothed, axis=0)
        nperseg = max(1, int(250.0*fs/1000.0)); noverlap = int(nperseg*0.95)
        freqs, times, Sxx = sp_signal.spectrogram(mean_sig, fs=fs, window="hann", nperseg=nperseg, noverlap=noverlap)
        fm = (freqs >= 1) & (freqs <= 100)
        if np.any(fm) and Sxx.size > 0:
            Sxx_f = Sxx[fm, :]
            Sxx_n = np.sqrt((Sxx_f - Sxx_f.min()) / (Sxx_f.max() - Sxx_f.min() + 1e-12))
            # Bicubic 4× upsample for smooth heatmap
            if Sxx_n.shape[0] > 2 and Sxx_n.shape[1] > 2:
                Sxx_n = zoom(Sxx_n, (4, 4), order=3)
                freqs_up = np.linspace(freqs[fm][0], freqs[fm][-1], Sxx_n.shape[0])
                times_up = np.linspace(times[0], times[-1], Sxx_n.shape[1])
            else:
                freqs_up = freqs[fm]; times_up = times
            fig.add_trace(go.Heatmap(z=Sxx_n, x=times_up*1000, y=freqs_up,
                colorscale="Jet", showscale=False, zsmooth="best"), row=2, col=1)
    except Exception: pass
    fig.update_yaxes(title_text="Freq (Hz)", row=2, col=1)
    # Panel 3: PSD
    mean_sig_full = np.mean(traces_data, axis=0); N = len(mean_sig_full); fs_full = 1000.0/dt
    fft_v = np.fft.rfft(mean_sig_full); f = np.fft.rfftfreq(N, d=dt/1000.0)
    psd_raw = np.abs(fft_v)**2 / (N*fs_full); mask = f <= 100
    psd_n = psd_raw[mask] / (np.max(psd_raw[mask]) + 1e-12)
    fig.add_trace(go.Scatter(x=f[mask], y=psd_n, mode="lines", line=dict(color="#81C784", width=2),
        fill="tozeroy", fillcolor="rgba(129,199,132,0.15)", showlegend=False), row=3, col=1)
    fig.update_yaxes(title_text="Norm Power", row=3, col=1)
    fig.update_xaxes(title_text="Frequency (Hz)", row=3, col=1)
    # Panel 4: Loss
    if has_loss:
        fig.add_trace(go.Scatter(x=list(range(len(report.loss_history))), y=report.loss_history,
            mode="lines", line=dict(color="#FFB74D", width=2), showlegend=False), row=4, col=1)
        fig.update_xaxes(title_text="Epoch", row=4, col=1)
        fig.update_yaxes(title_text="Loss", row=4, col=1)
    fig.update_layout(title=dict(text=title, font=dict(size=20, family="Inter")),
        template="plotly_dark", height=300*n_rows, margin=dict(l=60, r=20, t=80, b=50), showlegend=False)
    return fig
