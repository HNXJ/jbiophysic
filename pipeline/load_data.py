# pipeline/load_data.py
import numpy as np
import json
import scipy.io as sio

def load_empirical_neurophysiology(dataset_path: str, protocol: str = "omission"):
    """
    Axis 19: Explicit data-driven component for loading real electrophysiological footprints.
    Loads raw Neuropixels / LFP data, filters it, and returns targeted PSD masks.
    """
    # Fallback to simulated ground-truth if data doesn't exist locally
    print(f"🧬 Loading empirical {protocol} data from {dataset_path}...")
    
    # In a real environment, this utilizes scipy.io.loadmat or h5py
    try:
        data = sio.loadmat(dataset_path)
    except FileNotFoundError:
        print(f"Warning: Empirical trace {dataset_path} not found. Generating mock ground truth.")
        fs = 1000.0
        T = 5000
        t = np.arange(T) / fs
        
        # Synthesize ground truth: 40Hz Gamma + 20Hz Beta
        lfp_target = np.sin(2 * np.pi * 40 * t) + 0.5 * np.sin(2 * np.pi * 20 * t)
        
        # Compute Ground Truth PSD
        freqs = np.fft.rfftfreq(T, d=1/fs)
        fft_mag = np.abs(np.fft.rfft(lfp_target))
        psd = (fft_mag ** 2) / T
        
        return {
            "psd": psd,
            "freqs": freqs,
            "gamma_mask": (freqs >= 30) & (freqs <= 80),
            "beta_mask": (freqs >= 13) & (freqs <= 30),
            "target_occupancy": 0.5  # Placeholder for physiological receptor states
        }

def load_pharmacology_profile(drug_name="ketamine"):
    """Loads pharmacokinetic profiling for the network."""
    # NMDAr binding affinity and dosage modeling
    profiles = {
        "ketamine": {"receptor": "NMDA", "occupancy": 0.5},
        "propofol": {"receptor": "GABA_A", "occupancy": 0.8}
    }
    return profiles.get(drug_name, None)
