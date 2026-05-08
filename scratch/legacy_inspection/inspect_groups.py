import sys
import os
import jaxley as jx

# Add src to path to import jbiophysic
sys.path.insert(0, os.path.join(os.getcwd(), "src"))

cell = jx.Cell([jx.Branch(ncomp=1)], parents=[-1])
net = jx.Network([cell])

print(f"Network has add_to_group: {hasattr(net, 'add_to_group')}")
if hasattr(net, 'add_to_group'):
    print(f"type(net.add_to_group): {type(net.add_to_group)}")

try:
    net.add_to_group("PC", [0])
    print("Successfully added to group 'PC'")
    print(f"Group names: {net.group_names}")
except Exception as e:
    print(f"Failed to add to group: {e}")
