import jaxley as jx
import sys
import os
import pandas as pd

# Add project root to path to import SafeHH
sys.path.insert(0, os.getcwd())
from jbiophysics.core.mechanisms.models import SafeHH

comp = jx.Compartment()
cell = jx.Cell([comp], parents=[-1])
cell.insert(SafeHH(name="HH"))

print("Columns in cell.nodes_df:")
print(cell.nodes_df.columns.tolist())
