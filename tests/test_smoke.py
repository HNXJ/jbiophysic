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

if __name__ == "__main__":
    test_import()
    test_results_report()
    test_plotly_dashboard()
    test_netbuilder_api()
    print("\n🏁 All smoke tests passed!")
