import sys
import os
import jaxley as jx

# Add src to path to import jbiophysic
sys.path.insert(0, os.path.join(os.getcwd(), "src"))

from jbiophysic.core.mechanisms.channels.hh_base import HH

cell = jx.Cell([jx.Branch(ncomp=1)], parents=[-1])
hh_inst = HH()
cell.insert(hh_inst)

# Try to set a parameter to see what works
print("Attempting to set 'gk'...")
try:
    cell.set("gk", 0.04)
    print("Successfully set 'gk'")
except Exception as e:
    print(f"Failed to set 'gk': {e}")

print("\nAttempting to set 'gl'...")
try:
    cell.set("gl", 0.0002)
    print("Successfully set 'gl'")
except Exception as e:
    print(f"Failed to set 'gl': {e}")
