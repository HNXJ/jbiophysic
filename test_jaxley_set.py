import jaxley as jx
from jaxley.channels import HH
import sys

comp = jx.Compartment()
cell = jx.Cell([comp], parents=[-1])
cell.insert(HH(name="HH"))

try:
    cell.set("HH_gNa", 120.0)
    print("cell.set('HH_gNa') worked")
except Exception as e:
    print(f"cell.set('HH_gNa') failed: {type(e).__name__}: {e}")

try:
    cell.nodes().set("HH_gNa", 120.0)
    print("cell.nodes().set('HH_gNa') worked")
except Exception as e:
    print(f"cell.nodes().set('HH_gNa') failed: {type(e).__name__}: {e}")
