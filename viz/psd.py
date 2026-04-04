import numpy as np
from scipy import signal

def compute_psd(trace, dt, f_min=1, f_max=100):
    """Simple PSD wrapper for OptimizerFacade."""
    fs = 1000.0 / dt
    f, pxx = signal.welch(trace - np.mean(trace), fs=fs, nperseg=min(len(trace), int(fs)))
    mask = (f >= f_min) & (f <= f_max)
    return f[mask], pxx[mask]
