import jaxley as jx
import sys
import os

# Add project root to path to import SafeHH
sys.path.insert(0, os.getcwd())
try:
    from jbiophysics.core.mechanisms.models import SafeHH
    print("SafeHH imported successfully")
except ImportError as e:
    print(f"ImportError: {e}")
    sys.exit(1)

comp = jx.Compartment()
cell = jx.Cell([comp], parents=[-1])
cell.insert(SafeHH(name="HH"))

print("\nParameters available on cell after insert(SafeHH(name='HH')):")
print(cell.get_parameters())
