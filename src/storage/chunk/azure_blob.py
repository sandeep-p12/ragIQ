"""Azure Blob chunk store placeholder (TODO: implement for future migration)."""

from typing import Any, Dict, List, Optional

from src.core.interfaces import ChunkStore


class AzureBlobStore(ChunkStore):
    """Azure Blob Storage chunk store implementation (TODO: not yet implemented).
    
    This is a placeholder for future Azure Blob Storage integration.
    The interface matches ChunkStore protocol for easy migration.
    """
    
    def __init__(self, *args, **kwargs):
        """Initialize Azure Blob store (placeholder)."""
        raise NotImplementedError(
            "Azure Blob Storage integration is not yet implemented. "
            "This is a placeholder for future migration. "
            "Use LocalChunkStore for now."
        )
    
    def put_chunks(
        self,
        children: List[Dict[str, Any]],
        parents: List[Dict[str, Any]]
    ) -> None:
        """Store chunks (not implemented)."""
        raise NotImplementedError("Azure Blob put_chunks not implemented")
    
    def get_chunk(
        self,
        doc_id: str,
        chunk_id: str,
        is_parent: bool = False
    ) -> Optional[Dict[str, Any]]:
        """Get chunk (not implemented)."""
        raise NotImplementedError("Azure Blob get_chunk not implemented")
    
    def get_chunks_bulk(
        self,
        keys: List[tuple[str, str, bool]]
    ) -> List[Optional[Dict[str, Any]]]:
        """Get chunks bulk (not implemented)."""
        raise NotImplementedError("Azure Blob get_chunks_bulk not implemented")

