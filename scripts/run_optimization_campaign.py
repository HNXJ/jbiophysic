# scripts/run_optimization_campaign.py

import os
from dataclasses import asdict
from compose import OptimizerFacade
from networks.omission_two_column import OmissionTrialConfig, build_omission_network
from neurons.cortical import make_synapses_independent

# Ensure the output directory exists
output_dir = "./jbiophysics/results/campaign_260328_omission/optimization"
os.makedirs(output_dir, exist_ok=True)

# 1. Build the network using the standard, tested builder.
print("Building the two-column omission network...")
onet = build_omission_network(seed=42)
net = onet.net
print("Network built successfully.")

# 2. Make key parameters trainable using the correct, post-build function.
# This was the step that was previously implemented incorrectly.
print("Making network parameters trainable...")
make_synapses_independent(net, "gAMPA")
make_synapses_independent(net, "gGABAa")
net.select(nodes="all").make_trainable("amp_noise") # For Inoise mechanism
print("Parameters marked as trainable.")

# 3. Configure the Optimizer with our specific biological goals
config = OmissionTrialConfig(stim_amp=3.0, td_amp=0.8)

# Manually construct the {pop_name: (start, end)} map required by the optimizer.
# This corrects the previous error caused by passing lists of indices.
print("Constructing population offset map for the optimizer...")
pop_offsets_map = {}
combined_pops = {**asdict(onet.v1_pops), **asdict(onet.ho_pops)}
for pop_name, index_list in combined_pops.items():
    if index_list: # Ensure the list is not empty
        pop_offsets_map[pop_name] = (min(index_list), max(index_list) + 1)
print("Offset map constructed.")

print("Configuring the OptimizerFacade with AGSDR...")
facade = (
    OptimizerFacade(net, method="AGSDR", lr=1e-4)
    .set_pop_offsets(pop_offsets_map)
    .set_pop_constraints("v1_l23_pyr", firing_rate=(5.0, 15.0))
    .set_pop_constraints("v1_l4_pyr", firing_rate=(5.0, 15.0))
    .set_pop_constraints("v1_l56_pyr", firing_rate=(5.0, 15.0))
    .set_pop_constraints("v1_pv", firing_rate=(20.0, 70.0))
    .set_pop_constraints("ho_l23_pyr", firing_rate=(5.0, 15.0))
    .set_pop_constraints("ho_l56_pyr", firing_rate=(5.0, 15.0))
    .set_constraints(kappa_max=0.1)
)
print("Optimizer configured.")

# 4. Run the campaign
print(f"Starting optimization campaign for 500 epochs...")
optimized_report = facade.run(epochs=500, dt=0.025, t_max=2000.0)

# 5. Save the results
print(f"Optimization complete. Saving final report and parameters to {output_dir}")
optimized_report.export(formats=["json", "plotly"], output_dir=output_dir)
print("Campaign finished successfully.")
