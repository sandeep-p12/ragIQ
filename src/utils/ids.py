"""ID generation utilities."""

import hashlib
import json
from typing import Any, Dict


def generate_chunk_id(content: str, metadata: Dict[str, Any]) -> str:
    """Generate deterministic chunk ID using SHA256 hash.
    
    Args:
        content: Chunk content (raw_md_fragment)
        metadata: Chunk metadata dictionary
    
    Returns:
        str: 16-character hexadecimal hash
    """
    # Create a stable string representation
    metadata_str = json.dumps(metadata, sort_keys=True)
    combined = f"{content}:{metadata_str}"
    hash_obj = hashlib.sha256(combined.encode())
    return hash_obj.hexdigest()[:16]

