"""ID generation utilities"""

import uuid
from typing import Optional


def generate_id(prefix: Optional[str] = None) -> str:
    """
    Generate a unique ID.
    
    Args:
        prefix: Optional prefix for the ID
        
    Returns:
        Unique ID string
    """
    id_str = str(uuid.uuid4())
    if prefix:
        return f"{prefix}_{id_str}"
    return id_str

