import sys
import os
import jaxley as jx

# Add src to path to import jbiophysic
sys.path.insert(0, os.path.join(os.getcwd(), "src"))

try:
    from jbiophysic.core.mechanisms.channels.hh_base import HH
except ImportError as e:
    print(f"ImportError: {e}")
    # Try alternate location if not found
    try:
        from jbiophysic.core.mechanisms.channels.hh import HH
    except ImportError as e2:
        print(f"ImportError (alt): {e2}")
        sys.exit(1)

cell = jx.Cell([jx.Branch(ncomp=1)], parents=[-1])
hh_inst = HH()
cell.insert(hh_inst)

print(f"HH class type: {type(HH)}")
print(f"HH instance type: {type(hh_inst)}")
print(f"HH instance name: {getattr(hh_inst, 'name', 'N/A')}")

print("\nRelevant cell node columns:")
for col in cell.nodes.columns:
    if any(k in col for k in ["gK", "gLeak", "HH", "Na", "gNa"]):
        print(f"  {col}")
