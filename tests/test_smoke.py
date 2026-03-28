"""Smoke tests for jbiophysics package."""

def test_import():
    import jbiophysics as jbp
    assert hasattr(jbp, "NetBuilder")
    assert hasattr(jbp, "ResultsReport")
    assert hasattr(jbp, "SDR")
    assert hasattr(jbp, "GSDR")
    assert hasattr(jbp, "AGSDR")
    assert hasattr(jbp, "Inoise")
    assert hasattr(jbp, "GradedAMPA")
    print("✅ test_import passed")

def test_netbuilder_api():
    import jbiophysics as jbp
    net = (jbp.NetBuilder(seed=42)
        .add_population("E", n=5, cell_type="pyramidal")
        .add_population("I", n=3, cell_type="pv")
        .connect("E", "all", synapse="AMPA", p=0.3)
        .connect("I", "E", synapse="GABAa", p=0.5)
        .build())
    num_cells = len(list(net.cells))
    assert num_cells == 8, f"Expected 8, got {num_cells}"
    print(f"✅ test_netbuilder_api passed ({num_cells} cells)")

def test_results_report():
    import numpy as np
    from jbiophysics.export import ResultsReport
    report = ResultsReport(traces=np.random.randn(10, 1000)*30-65,
        loss_history=[1.0, 0.5, 0.3], dt=0.1, t_max=100.0, metadata={"test": True})
    d = report.to_dict()
    assert d["num_cells"] == 10
    assert len(d["loss_history"]) == 3
    print("✅ test_results_report passed")

def test_plotly_dashboard():
    import numpy as np
    from jbiophysics.export import ResultsReport
    from jbiophysics.viz.dashboard import generate_dashboard
    report = ResultsReport(traces=np.random.randn(10, 5000)*30-65,
        dt=0.1, t_max=500.0, loss_history=[1.0, 0.7, 0.4])
    fig = generate_dashboard(report)
    assert fig is not None
    print("✅ test_plotly_dashboard passed")

def test_v1_column_build():
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from systems.networks.omission_v1_column import build_v1_column
    net, pops = build_v1_column(seed=0)
    n = len(pops.all)
    assert n == 200, f"Expected 200, got {n}"
    assert len(pops.l4_pyr) == 40
    assert len(pops.pv) == 20
    print(f"✅ test_v1_column_build passed ({n} cells)")

def test_omission_network_build():
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from systems.networks.omission_two_column import build_omission_network
    onet = build_omission_network(seed=0)
    total = onet.n_v1 + onet.n_ho
    assert total == 300, f"Expected 300, got {total}"
    assert onet.n_v1 == 200
    assert onet.n_ho == 100
    print(f"✅ test_omission_network_build passed ({total} cells)")

def test_omission_metrics():
    import sys, os, numpy as np
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from core.optimizers.omission_metrics import (
        compute_rsa, compute_fleiss_kappa, compute_sss
    )
    # Dummy spike dicts
    sp1 = {0: np.array([10, 50, 100]), 1: np.array([20, 80])}
    sp2 = {0: np.array([12, 52, 98]),  1: np.array([22, 82])}
    n_steps = 400
    rsa = compute_rsa(sp1, sp2, [0, 1], n_steps)
    assert np.isfinite(rsa), "RSA NaN"
    kappa = compute_fleiss_kappa(sp1, {"L23": [0], "L4": [1]}, n_steps)
    assert np.isfinite(kappa), "Kappa NaN"
    lfp = np.sin(2*np.pi*10*np.arange(n_steps)*0.025/1000.0)
    sss = compute_sss(lfp, lfp, fs=40000.0)
    assert np.isfinite(sss), "SSS NaN"
    print(f"✅ test_omission_metrics passed (rsa={rsa:.3f} kappa={kappa:.3f} sss={sss:.3f})")

def test_omission_raster_viz():
    import sys, os, numpy as np
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from jbiophysics.viz.omission_viz import plot_omission_raster
    # Synthetic traces: 10 cells × 400 timesteps
    traces = np.random.randn(10, 400) * 5 - 65.0
    # Artificially inject a spike at t=50 for cell 0
    traces[0, 50] = 20.0
    pops = {"l23": [0, 1, 2], "l4": [3, 4, 5], "l56": [6, 7, 8, 9]}
    b64 = plot_omission_raster(traces, dt_ms=0.025, pops=pops)
    assert len(b64) > 100, "Base64 PNG unexpectedly short"
    print(f"✅ test_omission_raster_viz passed (b64 len={len(b64)})")

if __name__ == "__main__":

    test_import()
    test_results_report()
    test_plotly_dashboard()
    test_netbuilder_api()
    test_v1_column_build()
    test_omission_network_build()
    test_omission_metrics()
    test_omission_raster_viz()
    print("\n🏁 All smoke tests passed!")

