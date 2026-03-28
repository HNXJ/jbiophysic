import jax
import jax.numpy as jnp
import jaxley as jx
import numpy as np
import pandas as pd
from typing import Dict, List, Any
import time

def run_parameter_sweep(net, param_manifold: Dict[str, List[float]], simulation_func: Any, save_path: str):
    """
    Performs an automated parameter sweep over a grid defined in param_manifold.
    Calculates Mean AFR and Kappa for each point.
    """
    print(f"🚀 Starting Automated Parameter Sweep...")
    
    # 1. Generate Grid
    import itertools
    keys = list(param_manifold.keys())
    values = list(param_manifold.values())
    grid = list(itertools.product(*values))
    
    results = []
    
    for i, combination in enumerate(grid):
        current_params = dict(zip(keys, combination))
        print(f"  [{i+1}/{len(grid)}] Testing: {current_params}")
        
        # Apply parameters to network
        # This assumes parameters are top-level or view-level
        for key, val in current_params.items():
            setattr(net.edges, key, val)
        
        # 2. Simulate
        start_t = time.time()
        afr, kappa = simulation_func(net)
        duration = time.time() - start_t
        
        # 3. Store
        res_row = current_params.copy()
        res_row['mean_afr'] = float(afr)
        res_row['kappa'] = float(kappa)
        res_row['sim_duration_sec'] = duration
        results.append(res_row)

    # 4. Save
    df = pd.DataFrame(results)
    df.to_csv(save_path, index=False)
    print(f"✨ Sweep complete. Results saved to {save_path}")
    return df
