import pytest
import numpy as np
from jbiophysic.viz import jvis, JVis
try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

@pytest.mark.skipif(not MATPLOTLIB_AVAILABLE, reason="matplotlib not available")
def test_jvis_raster_real():
    spikes = np.zeros((100, 2))
    spikes[10, 0] = 1.0
    fig, ax = jvis.raster(spikes)
    assert fig is not None
    # Check that scatter plot was created (collections)
    assert len(ax.collections) > 0
    plt.close(fig)

@pytest.mark.skipif(not MATPLOTLIB_AVAILABLE, reason="matplotlib not available")
def test_jvis_traces_real():
    v = np.random.randn(100, 5)
    fig, ax = jvis.traces(v)
    assert fig is not None
    assert len(ax.lines) == 5
    plt.close(fig)

@pytest.mark.skipif(not MATPLOTLIB_AVAILABLE, reason="matplotlib not available")
def test_jvis_summary_real():
    data = {
        "spikes": np.zeros((100, 2)),
        "v": np.random.randn(100, 2),
        "lfp": np.random.randn(100)
    }
    fig, axes = jvis.summary(data)
    assert fig is not None
    assert axes.shape == (2, 2)
    plt.close(fig)
