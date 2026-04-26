# src/jbiophysic/common/utils/serialization.py
from jbiophysic.common.utils.logging import get_logger

logger = get_logger(__name__)

import json
import numpy as np
from typing import Any

def sanitize_nans(obj: Any) -> Any:
    """Recursively converts NaN and Inf values to None for JSON compliance."""
    if isinstance(obj, dict):
        return {k: sanitize_nans(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [sanitize_nans(v) for v in obj]
    elif isinstance(obj, tuple):
        return tuple(sanitize_nans(v) for v in obj)
    elif isinstance(obj, (float, np.floating)):
        if np.isnan(obj) or np.isinf(obj):
            return None
    elif isinstance(obj, (np.ndarray, np.generic)):
        return sanitize_nans(obj.tolist())
    return obj

def safe_serialize_json(data: Any, indent: int = 2) -> str:
    """Serializes data to JSON with NaN-to-null conversion."""
    logger.info("Executing safe JSON serialization")
    return json.dumps(sanitize_nans(data), indent=indent)

def safe_save_json(data: Any, filepath: str):
    """Saves data to JSON file with NaN-to-null conversion."""
    logger.info(f"Saving scientific JSON to {filepath}")
    with open(filepath, "w") as f:
        json.dump(sanitize_nans(data), f, indent=2)
    logger.info(f"✅ Data saved to {filepath}")
