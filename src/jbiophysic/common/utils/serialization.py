# src/jbiophysic/common/utils/serialization.py
from jbiophysic.common.utils.logging import get_logger

logger = get_logger(__name__)

import json
import numpy as np
from typing import Any

class ScientificJSONEncoder(json.JSONEncoder):
    """
    Standardizes null handling for scientific data.
    Prioritizes JSON null over NaN/Inf for cross-environment compatibility.
    """
    def default(self, obj: Any) -> Any:
        logger.info(f"Encoding object of type {type(obj)}")
        if isinstance(obj, (np.ndarray, np.generic)):
            logger.info("Detected numpy array/scalar; converting to list")
            return obj.tolist()
        if isinstance(obj, float) and (np.isnan(obj) or np.isinf(obj)):
            logger.info("Detected NaN or Inf; converting to JSON null")
            return None
        return super().default(obj)

def safe_serialize_json(data: Any, indent: int = 2) -> str:
    """Serializes data to JSON with NaN-to-null conversion."""
    logger.info("Executing safe JSON serialization")
    res = json.dumps(data, cls=ScientificJSONEncoder, indent=indent)
    return res

def safe_save_json(data: Any, filepath: str):
    """Saves data to JSON file with NaN-to-null conversion."""
    logger.info(f"Saving scientific JSON to {filepath}")
    with open(filepath, "w") as f:
        json.dump(data, f, cls=ScientificJSONEncoder, indent=2)
    logger.info(f"✅ Data saved to {filepath}")
