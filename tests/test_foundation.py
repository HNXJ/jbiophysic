import jax
import jaxley as jx
import numpy as np
from jbiophysics.compose import NetBuilder
from jbiophysics.core.mechanisms.models import SafeHH, Inoise

def test_safe_hh_naming():
    # Verify name is "HH" by default
    shh = SafeHH()
    assert shh.name == "HH"

def test_builder_synapse_normalization():
    builder = NetBuilder()
    builder.add_population("P1", n=1)
    builder.add_population("P2", n=1)
    # This should not raise KeyError with lower/upper
    builder.connect("P1", "P2", "AMPA", g=1.0)
    builder.connect("P1", "P2", "ampa", g=1.0)
    assert len(builder._connections) == 2

def test_v1_build():
    from jbiophysics.systems.networks.omission_v1_column import build_v1_column
    net, pops = build_v1_column(seed=1)
    assert len(pops.all) == 200
    print("Foundation tests passed")

if __name__ == "__main__":
    test_safe_hh_naming()
    test_builder_synapse_normalization()
    test_v1_build()
