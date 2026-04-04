import sys
import os
import numpy as np
import jax
import jax.numpy as jnp
import jaxley as jx
import optax

# Standard toolbox imports
from channels.hh import SafeHH, Inoise
from connect.graded import GradedAMPA, GradedGABAa
from neurons.cortical import build_pyramidal_cell, build_pv_cell
from networks.v1 import build_v1_column
from optimizers import SDR, GSDR, AGSDR
from compose import NetBuilder, OptimizerFacade
from utils.metrics import compute_rsa, compute_fleiss_kappa

def test_channels_and_neurons():
    print("Testing Channels and Neurons...")
    cell = build_pyramidal_cell()
    cell.insert(Inoise())
    net = jx.Network([cell])
    # Brief integration to check for NaNs or shape errors
    tr = jx.integrate(net, t_max=10.0, delta_t=0.1)
    assert not jnp.any(jnp.isnan(tr)), "NaN in channel integration"
    print("✅ Channels/Neurons OK")

def test_connectivity():
    print("Testing Connectivity...")
    c1, c2 = build_pyramidal_cell(), build_pv_cell()
    net = jx.Network([c1, c2])
    jx.connect(net.cell(0), net.cell(1), GradedAMPA(), p=1.0)
    # Check if param setting works (the SafeHH fix)
    net.cell(0).set("HH_gNa", 0.12) 
    print("✅ Connectivity/ParamSetting OK")

def test_netbuilder_fluent():
    print("Testing NetBuilder Fluent API...")
    builder = (NetBuilder(seed=123)
               .add_population("E", n=5, cell_type="pyramidal")
               .add_population("I", n=2, cell_type="pv")
               .connect("E", "I", synapse="AMPA", p=0.5)
               .connect("I", "E", synapse="GABAa", p=1.0)
               .build())
    assert len(builder.cell_types) >= 1, "Builder construction failed"
    print("✅ NetBuilder OK")

def test_optimizers_toy():
    print("Testing Optimizers (Toy Params)...")
    # Simple NetBuilder
    net = (NetBuilder(seed=42)
           .add_population("N", n=2, cell_type="pyramidal")
           .connect("all", "all", synapse="AMPA", p=1.0)
           .make_trainable("g")
           .build())
    
    # OptimizerFacade test (AGSDR)
    facade = OptimizerFacade(net, method="AGSDR", lr=0.1)
    facade.set_constraints(firing_rate=(1, 50))
    # Run a very short 5-epoch test
    res = facade.run(epochs=5, t_max=100.0, dt=0.1)
    assert len(res.loss_history) == 5, "Optimizer history failure"
    print("✅ Optimizers/Facade OK")

def test_metrics():
    print("Testing Metrics Utils...")
    # Toy spikes: {0: [10.0, 20.0], 1: [15.0]}
    sim = {0: [100, 200], 1: [150]} # indices in steps
    tgt = {0: [100, 200], 1: [150]}
    rsa = compute_rsa(sim, tgt, [0, 1], 1000)
    assert float(rsa) < 1e-6, f"RSA metric error for identical spikes: {rsa}"
    print("✅ Metrics OK")

def run_all_tests():
    print("🚀 Initializing Toolbox Suite Verification")
    print("-" * 40)
    try:
        test_channels_and_neurons()
        test_connectivity()
        test_netbuilder_fluent()
        test_optimizers_toy()
        test_metrics()
        print("-" * 40)
        print("🏆 All Toolbox functions passed verification!")
    except Exception as e:
        print("-" * 40)
        print(f"❌ TEST FAILED: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    run_all_tests()
