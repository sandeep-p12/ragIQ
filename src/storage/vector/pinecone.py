"""Pinecone vector store implementation."""

from typing import Any, Dict, List, Optional

try:
    from pinecone import Pinecone, ServerlessSpec
except ImportError:
    raise ImportError("pinecone is required. Install with: pip install pinecone")

from src.config.retrieval import PineconeConfig
from src.core.dataclasses import Candidate, VectorRecord
from src.core.interfaces import VectorStore


class PineconeVectorStore(VectorStore):
    """Pinecone vector store implementation."""
    
    def __init__(self, config: PineconeConfig = None):
        """Initialize Pinecone store.
        
        Args:
            config: PineconeConfig (defaults to PineconeConfig.from_env())
        """
        self.config = config or PineconeConfig.from_env()
        
        if not self.config.api_key:
            raise ValueError(f"{self.config.api_key_env} not found in environment")
        
        self.pc = Pinecone(api_key=self.config.api_key)
        self._ensure_index_exists()
        self.index = self.pc.Index(self.config.index_name)
    
    def _ensure_index_exists(self):
        """Create index if it doesn't exist."""
        existing_indexes = [idx.name for idx in self.pc.list_indexes()]
        
        if self.config.index_name not in existing_indexes:
            # Determine dimension based on embedding model
            # text-embedding-3-small = 1536, text-embedding-3-large = 3072
            dimension = 1536  # Default for text-embedding-3-small
            
            try:
                # Try to create serverless index (free tier)
                self.pc.create_index(
                    name=self.config.index_name,
                    dimension=dimension,
                    metric="cosine",
                    spec=ServerlessSpec(
                        cloud="aws",
                        region="us-east-1"
                    )
                )
            except Exception as e:
                # Fallback to pod-based if serverless fails
                try:
                    self.pc.create_index(
                        name=self.config.index_name,
                        dimension=dimension,
                        metric="cosine"
                    )
                except Exception as create_error:
                    raise ValueError(
                        f"Failed to create Pinecone index '{self.config.index_name}': {create_error}"
                    ) from create_error
    
    def upsert(
        self,
        vectors: List[VectorRecord],
        namespace: Optional[str] = None
    ) -> None:
        """Upsert vectors to Pinecone.
        
        Args:
            vectors: List of vector records
            namespace: Optional namespace (defaults to config namespace)
        """
        if not vectors:
            return
        
        ns = namespace or self.config.namespace
        
        # Convert VectorRecord to Pinecone format
        pinecone_vectors = []
        for vec in vectors:
            pinecone_vectors.append({
                "id": vec.id,
                "values": vec.values,
                "metadata": self._sanitize_metadata(vec.metadata)
            })
        
        # Upsert in batches (Pinecone recommends 100 at a time)
        batch_size = 100
        for i in range(0, len(pinecone_vectors), batch_size):
            batch = pinecone_vectors[i:i + batch_size]
            self.index.upsert(vectors=batch, namespace=ns)
    
    def query(
        self,
        vector: List[float],
        top_k: int,
        filters: Optional[Dict[str, Any]] = None,
        namespace: Optional[str] = None
    ) -> List[Candidate]:
        """Query Pinecone index.
        
        Args:
            vector: Query embedding vector
            top_k: Number of results to return
            filters: Optional metadata filters (e.g., {"doc_id": "doc1", "element_type": "table"})
            namespace: Optional namespace (defaults to config namespace)
            
        Returns:
            List of candidate chunks
        """
        ns = namespace or self.config.namespace
        
        # Build Pinecone filter
        pinecone_filter = None
        if filters:
            pinecone_filter = {}
            for key, value in filters.items():
                if isinstance(value, list):
                    pinecone_filter[key] = {"$in": value}
                else:
                    pinecone_filter[key] = {"$eq": value}
        
        # Query Pinecone
        response = self.index.query(
            vector=vector,
            top_k=top_k,
            include_metadata=True,
            namespace=ns,
            filter=pinecone_filter
        )
        
        # Convert to Candidate objects
        candidates = []
        for match in response.matches:
            candidates.append(Candidate(
                chunk_id=match.id,
                score=match.score or 0.0,
                metadata=match.metadata or {}
            ))
        
        return candidates
    
    def _sanitize_metadata(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Sanitize metadata for Pinecone (must be JSON-serializable).
        
        Args:
            metadata: Original metadata dict
            
        Returns:
            Sanitized metadata dict
        """
        sanitized = {}
        for key, value in metadata.items():
            # Pinecone metadata must be JSON-serializable
            if isinstance(value, (str, int, float, bool, list)):
                sanitized[key] = value
            elif isinstance(value, tuple):
                # Convert tuples to lists
                sanitized[key] = list(value)
            elif value is None:
                # Skip None values
                continue
            else:
                # Convert other types to string
                sanitized[key] = str(value)
        
        return sanitized

