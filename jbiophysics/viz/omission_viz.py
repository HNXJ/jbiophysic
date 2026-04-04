import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import io
import base64

# Colors from instructions
BG = "#0D0D0F"
GOLD = "#CFB87C"
WHITE = "#E8E8E8"
CYAN = "#00FFFF"
VIOLET = "#9400D3"

def plot_to_base64():
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100)
    plt.close()
    return base64.b64encode(buf.getvalue()).decode('utf-8')

def plot_omission_raster(traces, dt, pops_dict, title="Raster Plot"):
    plt.figure(figsize=(10, 6))
    plt.style.use('dark_background')
    plt.gca().set_facecolor(BG)
    
    offset = 0
    colors = [CYAN, GOLD, VIOLET, "red", "green", "orange", "blue"]
    
    threshold = -20.0
    for i, (pop_name, indices) in enumerate(pops_dict.items()):
        color = colors[i % len(colors)]
        for idx in indices:
            v = traces[idx]
            spikes = np.where((v[:-1] < threshold) & (v[1:] >= threshold))[0]
            plt.scatter(spikes * dt, np.ones_like(spikes) * offset, color=color, s=2, alpha=0.6)
            offset += 1
            
    plt.xlabel("Time (ms)")
    plt.ylabel("Neuron Index")
    plt.title(title, color=GOLD)
    return plot_to_base64()

def plot_lfp_traces(lfp_v1, lfp_ho, dt, title="LFP Traces"):
    plt.figure(figsize=(10, 4))
    plt.style.use('dark_background')
    plt.gca().set_facecolor(BG)
    
    t = np.arange(len(lfp_v1)) * dt
    plt.plot(t, lfp_v1, color=CYAN, label="V1 LFP")
    plt.plot(t, lfp_ho, color=VIOLET, label="HO LFP")
    plt.legend()
    plt.xlabel("Time (ms)")
    plt.ylabel("Voltage (mV)")
    plt.title(title, color=GOLD)
    return plot_to_base64()

def plot_tfr(lfp, dt, f_min=4, f_max=100, title="Time-Frequency Representation"):
    """Generate a spectrogram."""
    fs = 1000.0 / dt
    f, t, Sxx = signal.spectrogram(lfp, fs=fs, nperseg=int(fs/4), noverlap=int(fs/8))
    
    plt.figure(figsize=(10, 4))
    plt.style.use('dark_background')
    
    mask = (f >= f_min) & (f <= f_max)
    plt.pcolormesh(t * 1000.0, f[mask], 10 * np.log10(Sxx[mask, :]), shading='gouraud', cmap='magma')
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [ms]')
    plt.title(title, color=GOLD)
    plt.colorbar(label='dB/Hz')
    return plot_to_base64()
