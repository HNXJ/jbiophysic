import pytest
from jbiophysic.viz import jvis, JVis
try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

@pytest.mark.skipif(not MATPLOTLIB_AVAILABLE, reason="matplotlib not available")
def test_jvis_raster_returns_fig():
    fig, ax = jvis.raster(None)
    assert fig is not None
    assert ax is not None
    plt.close(fig)

@pytest.mark.skipif(not MATPLOTLIB_AVAILABLE, reason="matplotlib not available")
def test_jvis_summary_returns_fig():
    fig, axes = jvis.summary(None)
    assert fig is not None
    assert axes.shape == (2, 2)
    plt.close(fig)

@pytest.mark.skipif(not MATPLOTLIB_AVAILABLE, reason="matplotlib not available")
def test_jvis_wrapper():
    jv = JVis(None)
    fig, ax = jv.raster()
    assert fig is not None
    plt.close(fig)
