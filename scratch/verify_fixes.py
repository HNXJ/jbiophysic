import jax.numpy as jnp
import jaxley as jx
from jbiophysic.core.mechanisms.channels.hh_base import HH
from jbiophysic.core.mechanisms.synapses.kinetics import SpikingNMDA
from jbiophysic.models.builders.populations import build_pyramidal_cell, build_interneuron

def verify_hh():
    print("Verifying HH singularities...")
    hh = HH()
    dt = 0.1
    params = hh.channel_params
    states = hh.channel_states
    
    # Check at exactly -40.0
    v = jnp.array([-40.0, -55.0, -65.0])
    new_states = hh.update_states(states, dt, v, params)
    
    for k, val in new_states.items():
        if jnp.any(jnp.isnan(val)):
            print(f"FAILED: HH state {k} contains NaNs at singular points!")
            return False
    print("SUCCESS: HH states are finite at singular points.")
    return True

def verify_synapse():
    print("Verifying Synapse current selection...")
    nmda = SpikingNMDA(None, None)
    states = {"w": jnp.array([1.0, 1.0]), "s": jnp.array([1.0, 1.0])}
    params = nmda.synapse_params
    params["is_nmda"] = 1.0
    
    v_pre = jnp.array([-65.0, 0.0])
    v_post = jnp.array([-65.0, -20.0])
    
    try:
        i = nmda.compute_current(states, v_pre, v_post, params)
        print(f"SUCCESS: Synapse current computed with shape {i.shape}")
    except Exception as e:
        print(f"FAILED: Synapse current computation failed with {e}")
        return False
    return True

def verify_morphology():
    print("Verifying cell morphology construction...")
    try:
        pc = build_pyramidal_cell()
        print(f"SUCCESS: Pyramidal cell built with {len(pc.branches)} branches")
        pv = build_interneuron("PV")
        print(f"SUCCESS: Interneuron built with {len(pv.branches)} branches")
    except Exception as e:
        print(f"FAILED: Cell construction failed with {e}")
        return False
    return True

if __name__ == "__main__":
    v1 = verify_hh()
    v2 = verify_synapse()
    v3 = verify_morphology()
    
    if v1 and v2 and v3:
        print("\nALL VERIFICATIONS PASSED.")
    else:
        exit(1)
