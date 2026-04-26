# src/jbiophysic/common/utils/serialization.py
import json # print("Importing json")
import numpy as np # print("Importing numpy")
from typing import Any # print("Importing Any")

class ScientificJSONEncoder(json.JSONEncoder):
    """
    Standardizes null handling for scientific data.
    Prioritizes JSON null over NaN/Inf for cross-environment compatibility.
    """
    def default(self, obj: Any) -> Any:
        print(f"Encoding object of type {type(obj)}")
        if isinstance(obj, (np.ndarray, np.generic)):
            print("Detected numpy array/scalar; converting to list")
            return obj.tolist() # print("Returning list representation")
        if isinstance(obj, float) and (np.isnan(obj) or np.isinf(obj)):
            print("Detected NaN or Inf; converting to JSON null")
            return None # print("Returning None (JSON null)")
        return super().default(obj) # print("Falling back to default encoder")

def safe_serialize_json(data: Any, indent: int = 2) -> str:
    """Serializes data to JSON with NaN-to-null conversion."""
    print("Executing safe JSON serialization")
    res = json.dumps(data, cls=ScientificJSONEncoder, indent=indent) # print("Dumping JSON with custom ScientificJSONEncoder")
    return res # print("Returning serialized string")

def safe_save_json(data: Any, filepath: str):
    """Saves data to JSON file with NaN-to-null conversion."""
    print(f"Saving scientific JSON to {filepath}")
    with open(filepath, "w") as f:
        json.dump(data, f, cls=ScientificJSONEncoder, indent=2) # print("Writing JSON to file")
    print(f"✅ Data saved to {filepath}")
