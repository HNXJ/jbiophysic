from typing import Any

import numpy as np

try:
    import matplotlib.pyplot as plt
    from scipy import signal as sp_signal
except ImportError:
    plt = None
    sp_signal = None


class JVis:
    """Unified visualization wrapper for simulation results."""
    def __init__(self, result_or_data: Any):
        self.data = result_or_data

    def raster(self, **kwargs):
        return raster(self.data, **kwargs)

    def psd(self, **kwargs):
        return psd(self.data, **kwargs)

    def spectrogram(self, **kwargs):
        return spectrogram(self.data, **kwargs)

    def traces(self, **kwargs):
        return traces(self.data, **kwargs)

    def lfp(self, **kwargs):
        return lfp(self.data, **kwargs)

    def summary(self, **kwargs):
        return summary(self.data, **kwargs)

def _get_fig_ax(fig=None, ax=None):
    if plt is None:
        raise ImportError("matplotlib is required for jvis.")
    if fig is None or ax is None:
        fig, ax = plt.subplots()
    return fig, ax

def _extract_array(data, keys, attr=None):
    if isinstance(data, (np.ndarray, list)):
        return np.asarray(data)
    if isinstance(data, dict):
        for k in keys:
            if k in data:
                return np.asarray(data[k])
    if hasattr(data, "outputs") and isinstance(data.outputs, dict):
        for k in keys:
            if k in data.outputs:
                return np.asarray(data.outputs[k])
    if attr and hasattr(data, attr):
        return np.asarray(getattr(data, attr))
    return None

def raster(result_or_spikes, fig=None, ax=None, **kwargs):
    fig, ax = _get_fig_ax(fig, ax)
    spikes = _extract_array(result_or_spikes, ["spikes", "spike_matrix"], "spikes")
    if spikes is not None:
        t_idx, n_idx = np.where(spikes > 0)
        ax.scatter(t_idx, n_idx, s=1, c='black', marker='|')
        ax.set_xlabel("Time (steps)")
        ax.set_ylabel("Neuron Index")
    else:
        ax.text(0.5, 0.5, "No spike data found", ha='center')
    return fig, ax

def traces(result_or_traces, fig=None, ax=None, max_n=10, **kwargs):
    fig, ax = _get_fig_ax(fig, ax)
    v = _extract_array(result_or_traces, ["v", "voltage", "traces"], "v_trace")
    if v is not None:
        if v.ndim == 1:
            ax.plot(v)
        else:
            n_plot = min(v.shape[1] if v.shape[1] < v.shape[0] else v.shape[0], max_n)
            # Assume time is longer dimension
            if v.shape[0] < v.shape[1]:
                 ax.plot(v[:n_plot, :].T)
            else:
                 ax.plot(v[:, :n_plot])
        ax.set_xlabel("Time (steps)")
        ax.set_ylabel("Amplitude")
    else:
        ax.text(0.5, 0.5, "No trace data found", ha='center')
    return fig, ax

def psd(result_or_signal, fs=1000.0, fig=None, ax=None, **kwargs):
    fig, ax = _get_fig_ax(fig, ax)
    if sp_signal is None:
        raise ImportError("scipy is required for psd.")
    
    sig = _extract_array(result_or_signal, ["lfp", "signal", "spikes"], "lfp")
    if sig is not None:
        if sig.ndim > 1:
            sig = np.mean(sig, axis=1 if sig.shape[1] < sig.shape[0] else 0)
        f, pxx = sp_signal.welch(sig, fs=fs)
        ax.semilogy(f, pxx)
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("PSD")
    else:
        ax.text(0.5, 0.5, "No signal data found", ha='center')
    return fig, ax

def spectrogram(result_or_signal, fs=1000.0, fig=None, ax=None, **kwargs):
    fig, ax = _get_fig_ax(fig, ax)
    if sp_signal is None:
        raise ImportError("scipy is required for spectrogram.")
    
    sig = _extract_array(result_or_signal, ["lfp", "signal", "spikes"], "lfp")
    if sig is not None:
        if sig.ndim > 1:
            sig = np.mean(sig, axis=1 if sig.shape[1] < sig.shape[0] else 0)
        f, t, sxx = sp_signal.spectrogram(sig, fs=fs)
        ax.pcolormesh(t, f, 10 * np.log10(sxx + 1e-12), shading='gouraud')
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Frequency (Hz)")
    else:
        ax.text(0.5, 0.5, "No signal data found", ha='center')
    return fig, ax

def lfp(result_or_signal, fig=None, ax=None, **kwargs):
    return traces(result_or_signal, fig=fig, ax=ax, **kwargs)

def summary(result, **kwargs):
    if plt is None:
        raise ImportError("matplotlib is required for jvis.")
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    raster(result, fig=fig, ax=axes[0, 0])
    traces(result, fig=fig, ax=axes[0, 1])
    psd(result, fig=fig, ax=axes[1, 0])
    spectrogram(result, fig=fig, ax=axes[1, 1])
    fig.tight_layout()
    return fig, axes

# Global jvis object for functional API
class JVisFunctional:
    raster = staticmethod(raster)
    psd = staticmethod(psd)
    spectrogram = staticmethod(spectrogram)
    traces = staticmethod(traces)
    lfp = staticmethod(lfp)
    summary = staticmethod(summary)

jvis = JVisFunctional()
