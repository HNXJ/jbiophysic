import jaxley as jx
import numpy as np
import sys

comp = jx.Compartment()
cell = jx.Cell([comp], parents=[-1])
net = jx.Network([cell])

t_steps = 100
# Test shape (T, 1)
try:
    stim = np.ones((t_steps, 1))
    net.cell(0).branch(0).loc(0.0).stimulate(stim)
    print("stimulate((T, 1)) worked")
except Exception as e:
    print(f"stimulate((T, 1)) failed: {e}")

# Test shape (1, T)
try:
    stim = np.ones((1, t_steps))
    net.cell(0).branch(0).loc(0.0).stimulate(stim)
    print("stimulate((1, T)) worked")
except Exception as e:
    print(f"stimulate((1, T)) failed: {e}")

# Test shape (T,)
try:
    stim = np.ones((t_steps,))
    net.cell(0).branch(0).loc(0.0).stimulate(stim)
    print("stimulate((T,)) worked")
except Exception as e:
    print(f"stimulate((T,)) failed: {e}")
