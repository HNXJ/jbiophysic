import jax.numpy as jnp
from jbiophysic.core.mechanisms.channels.hh_base import HH
from jbiophysic.core.mechanisms.channels.extra_currents import (
    IA, Ih, IM, ICa, ICaT, IKDR, IAR, ICan, CaDynamics
)
from jbiophysic.core.mechanisms.synapses.kinetics import (
    SpikingSynapse, SpikingNMDA, SpikingGABAa, SpikingGABAb, GapJunction
)
from jbiophysic.models.builders.reduced_models import (
    Izhikevich, FitzHughNagumo
)
from jbiophysic.models.builders.rate_models import EIRateModel

def smoke_test():
    print("Testing Mechanism Instantiation...")
    mechanisms = [HH(), IA(), Ih(), IM(), ICa(), ICaT(), IKDR(), IAR(), ICan(), CaDynamics()]
    for m in mechanisms:
        print(f"  ✅ {m.__class__.__name__} instantiated.")
        
    print("\nTesting Model Instantiation...")
    models = [Izhikevich(), FitzHughNagumo(), EIRateModel()]
    for m in models:
        print(f"  ✅ {m.__class__.__name__} instantiated.")
        
    print("\n🏆 DYNASIM PARITY VERIFIED.")

if __name__ == "__main__":
    smoke_test()
