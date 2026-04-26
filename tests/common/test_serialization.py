# tests/common/test_serialization.py
import numpy as np
import json
from jbiophysic.common.utils.serialization import safe_serialize_json

def test_nan_to_null_conversion():
    print("🧪 Testing NaN to null JSON conversion")
    
    # 1. Prepare dummy trace data with problematic values
    data = {
        "neuron_id": 101,
        "membrane_potential": [ -65.0, -40.0, np.nan, -70.0 ],
        "conductance": np.array([ 0.1, 0.5, np.inf, 0.2 ]),
        "meta": { "status": "completed", "error_code": np.nan }
    }
    
    # 2. Serialize using the new utility
    json_str = safe_serialize_json(data)
    
    print("--- Serialized JSON Output ---")
    print(json_str)
    print("------------------------------")
    
    # 3. Validation checks
    assert "null" in json_str
    assert "NaN" not in json_str
    assert "Infinity" not in json_str
    
    print("✅ Serialization validation successful: NaNs and Infs correctly converted to null.")

if __name__ == "__main__":
    test_nan_to_null_conversion()
