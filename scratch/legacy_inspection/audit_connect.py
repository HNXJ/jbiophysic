import jaxley as jx
from jaxley.synapses import IonotropicSynapse

# pc has 2 comps, sst has 1 comp
pc = jx.Cell([jx.Branch(ncomp=1), jx.Branch(ncomp=1)], parents=[-1, 0])
sst = jx.Cell([jx.Branch(ncomp=1)], parents=[-1])
net = jx.Network([pc, sst])

print(f"PC ncomps: {len(net.cell(0).nodes)}")
print(f"SST ncomps: {len(net.cell(1).nodes)}")

try:
    jx.connect(net.cell(0), net.cell(1), IonotropicSynapse())
    print("Cell-level success")
except Exception as e:
    print(f"Cell-level fail: {e}")

try:
    jx.connect(net.cell(0).comp(0), net.cell(1).comp(0), IonotropicSynapse())
    print("Comp-level success")
except Exception as e:
    print(f"Comp-level fail: {e}")
