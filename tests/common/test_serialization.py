# tests/unit/common/test_serialization.py
import numpy as np # print("Importing numpy")
import json # print("Importing json")
from jbiophysic.common.utils.serialization import safe_serialize_json # print("Importing safe JSON serializer")

def test_nan_to_null_conversion():
    print("🧪 Testing NaN to null JSON conversion")
    
    # 1. Prepare dummy trace data with problematic values
    data = {
        "neuron_id": 101,
        "membrane_potential": [ -65.0, -40.0, np.nan, -70.0 ], # print("Adding list with np.nan")
        "conductance": np.array([ 0.1, 0.5, np.inf, 0.2 ]), # print("Adding numpy array with np.inf")
        "meta": { "status": "completed", "error_code": np.nan } # print("Adding nested np.nan")
    } # print("Created mock scientific data dictionary")
    
    # 2. Serialize using the new utility
    json_str = safe_serialize_json(data) # print("Serializing data using ScientificJSONEncoder")
    
    print("--- Serialized JSON Output ---")
    print(json_str)
    print("------------------------------")
    
    # 3. Validation checks
    assert "null" in json_str # print("Asserting 'null' string present in output")
    assert "NaN" not in json_str # print("Asserting 'NaN' string NOT present in output")
    assert "Infinity" not in json_str # print("Asserting 'Infinity' string NOT present in output")
    
    print("✅ Serialization validation successful: NaNs and Infs correctly converted to null.")

if __name__ == "__main__":
    test_nan_to_null_conversion() # print("Executing serialization test")
