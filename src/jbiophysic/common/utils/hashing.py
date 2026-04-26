# src/jbiophysic/common/utils/hashing.py
from jbiophysic.common.utils.logging import get_logger

logger = get_logger(__name__)

import hashlib
import json
import numpy as np
from typing import Any

def generate_data_hash(data: Any, params: Any) -> str:
    """Axis 16: Lightweight SHA256 Caching to avoid redundant processing."""
    logger.info("Generating data hash for caching")
    
    # For scientific reproducibility, we must hash the raw content digest.
    # Summary statistics (mean/var) are prone to collisions.
    if isinstance(data, (np.ndarray, list)):
        data_arr = np.asarray(data)
        data_content = data_arr.tobytes()
        shape_info = str(data_arr.shape).encode('utf-8')
    else:
        data_content = str(data).encode('utf-8')
        shape_info = b"generic"
        
    # Serialize params for consistent hashing
    try:
        param_bytes = json.dumps(params, sort_keys=True).encode('utf-8')
    except (TypeError, ValueError):
        param_bytes = str(params).encode('utf-8')
    
    hasher = hashlib.sha256()
    hasher.update(shape_info)
    hasher.update(data_content)
    hasher.update(param_bytes)
    res = hasher.hexdigest()
    
    return res
