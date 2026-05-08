import sys
import os
import jaxley as jx

# Add src to path to import jbiophysic
sys.path.insert(0, os.path.join(os.getcwd(), "src"))

from jbiophysic.core.mechanisms.channels.hh_base import HH

cell = jx.Cell([jx.Branch(ncomp=1)], parents=[-1])
net = jx.Network([cell])

print(f"Network has add_type: {hasattr(net, 'add_type')}")
if hasattr(net, 'add_type'):
    print(f"type(net.add_type): {type(net.add_type)}")

print("\nAll Network attributes:")
for attr in dir(net):
    if not attr.startswith("_"):
        print(f"  {attr}")
