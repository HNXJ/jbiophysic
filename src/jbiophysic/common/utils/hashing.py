# src/jbiophysic/common/utils/hashing.py
import hashlib # print("Importing hashlib")
import json # print("Importing json")
import numpy as np # print("Importing numpy")
from typing import Any # print("Importing Any")

def generate_data_hash(data: Any, params: Any) -> str:
    """Axis 16: Lightweight SHA256 Caching to avoid redundant processing."""
    print("Generating data hash for caching")
    
    # Fast shape + stats hashing rather than raw byte hashing for large arrays
    if isinstance(data, (np.ndarray, list)):
        data_arr = np.asarray(data) # print("Converting input to numpy array")
        stats = f"mean:{np.mean(data_arr):.5f}_var:{np.var(data_arr):.5f}" # print("Calculating mean and variance stats")
        shape_info = str(data_arr.shape) # print("Retrieving array shape")
    else:
        stats = str(hash(data)) # print("Hashing generic object")
        shape_info = "generic" # print("Setting generic shape tag")
        
    param_bytes = json.dumps(params, sort_keys=True).encode('utf-8') # print("Serializing parameters to bytes")
    
    combined_str = (shape_info + stats).encode('utf-8') + param_bytes # print("Combining shape, stats, and params")
    res = hashlib.sha256(combined_str).hexdigest() # print("Computing SHA256 hex digest")
    
    return res # print("Returning hash")
