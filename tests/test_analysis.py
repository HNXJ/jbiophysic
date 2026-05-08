import numpy as np

from jbiophysic.analysis.fano import fano_factor
from jbiophysic.analysis.lfp import csd_layer_summary, lfp_rms
from jbiophysic.analysis.spectra import band_power, beta_gamma_ratio, power_spectrum
from jbiophysic.analysis.spikes import firing_rate_hz, threshold_crossings


def test_spike_and_fano_statistics():
    v = np.array([-1.0, 0.2, -0.1, 0.5, -0.2])
    spikes = threshold_crossings(v, threshold=0.0)
    assert spikes.tolist() == [False, True, False, True, False]
    assert firing_rate_hz(spikes, dt_s=0.001) == 400.0
    assert fano_factor(np.array([1, 2, 3, 4]), ddof=0) == 0.5


def test_spectral_beta_gamma_helpers():
    fs = 1000.0
    t = np.arange(0.0, 1.0, 1.0 / fs)
    x = np.sin(2 * np.pi * 20.0 * t) + 0.1 * np.sin(2 * np.pi * 60.0 * t)
    f, p = power_spectrum(x, fs, nperseg=256)
    beta = band_power(f, p, (13.0, 30.0))
    gamma = band_power(f, p, (40.0, 100.0))
    assert beta > gamma
    assert beta_gamma_ratio(f, p) > 1.0


def test_lfp_summaries():
    csd = np.ones((4, 5, 6))
    assert csd_layer_summary(csd, axis=2).shape == (6,)
    assert float(lfp_rms(np.array([3.0, 4.0]))) == np.sqrt(12.5)
