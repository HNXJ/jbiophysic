import sys
import os
import jaxley as jx

# Add src to path to import jbiophysic
sys.path.insert(0, os.path.join(os.getcwd(), "src"))

from jbiophysic.core.mechanisms.channels.hh_base import HH

cell = jx.Cell([jx.Branch(ncomp=1)], parents=[-1])
hh_inst = HH()
cell.insert(hh_inst)

print(f"HH instance name: {hh_inst.name}")

print("\nAll cell node columns:")
for col in cell.nodes.columns:
    print(f"  {col}")

# Try to set a parameter to see what works
try:
    cell.set("HH_gk", 0.04)
    print("\nSuccessfully set HH_gk")
except Exception as e:
    print(f"\nFailed to set HH_gk: {e}")

try:
    cell.set("HH_gk", 0.04)
    print("Successfully set HH_gk")
except Exception as e:
    print(f"Failed to set HH_gk: {e}")

