import jaxley as jx
from jaxley.synapses import IonotropicSynapse
from jbiophysic.models.builders.populations import construct_column
from typing import Dict, Any

class ExcSynapse(IonotropicSynapse): pass
class InhSynapse(IonotropicSynapse): pass

def build_v1_column_jaxley(params: Dict[str, float]) -> jx.Network:
    """
    Instantiates PC, PV, SST, VIP populations using the shared builder.
    Configures a PING-style gamma network using built-in HH mechanisms.
    """
    net = construct_column()
    
    # 1. Parameter Extraction
    pv_gain = params.get("pv_gain", 1.0)
    pc_to_pv_w = params.get("pc_to_pv_w", 0.001)
    pv_to_pc_w = params.get("pv_to_pc_w", 0.005)
    pv_to_pv_w = params.get("pv_to_pv_w", 0.002)
    drive_amp = params.get("drive_amp", 0.1) # stimulus amplitude in nA
    
    # 2. PV Modulation (using built-in HH prefix)
    base_gk = 0.036 * 1.5
    net.PV.set("HH_gK", base_gk * pv_gain)
    
    # 3. Connectivity
    # Use sparse_connect for population-level wiring
    syn_e = ExcSynapse()
    syn_i = InhSynapse()
    
    # PC -> PV (Exc)
    jx.sparse_connect(net.PC, net.PV, syn_e, p=0.1)
    # PV -> PC (Inh)
    jx.sparse_connect(net.PV, net.PC, syn_i, p=0.15)
    # PV -> PV (Inh)
    jx.sparse_connect(net.PV, net.PV, syn_i, p=0.1)
    
    # 4. Set weights and reversal potentials
    net.set("ExcSynapse_gS", pc_to_pv_w)
    net.set("ExcSynapse_e_syn", 0.0)
    net.set("InhSynapse_gS", pv_to_pc_w)
    net.set("InhSynapse_e_syn", -75.0)
    
    # 5. Stimulus Setup (0-100 baseline, 100-300 evoked)
    current = jx.step_current(i_delay=100.0, i_dur=200.0, i_amp=drive_amp, delta_t=0.025, t_max=300.0)
    net.PC.branch(0).loc(0.5).stimulate(current)
    
    # 6. Recording (population mean for LFP proxy)
    net.PC.record("v")
    
    return net
