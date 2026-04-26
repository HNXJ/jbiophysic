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
    
    # Fast shape + stats hashing rather than raw byte hashing for large arrays
    if isinstance(data, (np.ndarray, list)):
        data_arr = np.asarray(data)
        stats = f"mean:{np.mean(data_arr):.5f}_var:{np.var(data_arr):.5f}"
        shape_info = str(data_arr.shape)
    else:
        stats = str(hash(data))
        shape_info = "generic"
        
    param_bytes = json.dumps(params, sort_keys=True).encode('utf-8')
    
    combined_str = (shape_info + stats).encode('utf-8') + param_bytes
    res = hashlib.sha256(combined_str).hexdigest()
    
    return res
