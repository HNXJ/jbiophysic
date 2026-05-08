from typing import Any
try:
    from scipy import signal as sp_signal
except ImportError:
    sp_signal = None

def psd(signal: Any, fs: float):
    """Compute the Power Spectral Density (PSD)."""
    if sp_signal is None:
        raise ImportError("scipy is required for spectra ops.")
    return sp_signal.welch(signal, fs=fs)

def spectrogram(signal: Any, fs: float):
    """Compute the spectrogram."""
    if sp_signal is None:
        raise ImportError("scipy is required for spectra ops.")
    return sp_signal.spectrogram(signal, fs=fs)
