import jaxley as jx
from jaxley.synapses import IonotropicSynapse
import pandas as pd

# pc has 2 comps, sst has 1 comp
pc = jx.Cell([jx.Branch(ncomp=1), jx.Branch(ncomp=1)], parents=[-1, 0])
sst = jx.Cell([jx.Branch(ncomp=1)], parents=[-1])
net = jx.Network([pc, sst])

pre = net.cell(0).comp(0)
post = net.cell(1).comp(0)

print(f"Pre nodes length: {len(pre.nodes)}")
print(f"Post nodes length: {len(post.nodes)}")
print(f"Pre nodes index: {pre.nodes.index.tolist()}")
print(f"Post nodes index: {post.nodes.index.tolist()}")

try:
    jx.connect(pre, post, IonotropicSynapse())
    print("Connect success")
except Exception as e:
    print(f"Connect fail: {e}")
    import traceback
    traceback.print_exc()
