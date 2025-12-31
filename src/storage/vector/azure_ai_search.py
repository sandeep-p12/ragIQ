"""Azure AI Search vector store placeholder (TODO: implement for future migration)."""

from typing import Any, Dict, List, Optional

from src.core.dataclasses import Candidate, VectorRecord
from src.core.interfaces import VectorStore


class AzureAISearchStore(VectorStore):
    """Azure AI Search vector store implementation (TODO: not yet implemented).
    
    This is a placeholder for future Azure AI Search integration.
    The interface matches VectorStore protocol for easy migration.
    """
    
    def __init__(self, *args, **kwargs):
        """Initialize Azure AI Search store (placeholder)."""
        raise NotImplementedError(
            "Azure AI Search integration is not yet implemented. "
            "This is a placeholder for future migration. "
            "Use PineconeVectorStore for now."
        )
    
    def upsert(
        self,
        vectors: List[VectorRecord],
        namespace: Optional[str] = None
    ) -> None:
        """Upsert vectors (not implemented)."""
        raise NotImplementedError("Azure AI Search upsert not implemented")
    
    def query(
        self,
        vector: List[float],
        top_k: int,
        filters: Optional[Dict[str, Any]] = None,
        namespace: Optional[str] = None
    ) -> List[Candidate]:
        """Query vectors (not implemented)."""
        raise NotImplementedError("Azure AI Search query not implemented")

