"""Pinecone vector store implementation."""

import logging
from typing import Any, Dict, List, Optional

try:
    from pinecone import Pinecone, ServerlessSpec
except ImportError:
    raise ImportError("pinecone is required. Install with: pip install pinecone")

from src.config.retrieval import PineconeConfig
from src.core.dataclasses import Candidate, VectorRecord
from src.core.interfaces import VectorStore

logger = logging.getLogger(__name__)


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
        
        # Log index stats on initialization to monitor record count
        try:
            stats = self.index.describe_index_stats()
            total_vectors = stats.get('total_vector_count', 0)
            namespaces = stats.get('namespaces', {})
            logger.info(f"Pinecone index '{self.config.index_name}' initialized. Total vectors: {total_vectors}")
            for ns_name, ns_stats in namespaces.items():
                ns_count = ns_stats.get('vector_count', 0)
                logger.info(f"  Namespace '{ns_name}': {ns_count} vectors")
        except Exception as e:
            logger.warning(f"Could not retrieve index stats: {e}")
    
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
            logger.warning("No vectors provided to upsert, skipping")
            return
        
        ns = namespace or self.config.namespace
        logger.info(f"Upserting {len(vectors)} vectors to Pinecone index '{self.config.index_name}' in namespace '{ns}'")
        
        # Convert VectorRecord to Pinecone format
        pinecone_vectors = []
        for vec in vectors:
            if not vec.id:
                logger.warning(f"Skipping vector with empty ID: {vec}")
                continue
            if not vec.values:
                logger.warning(f"Skipping vector {vec.id} with empty values")
                continue
            pinecone_vectors.append({
                "id": vec.id,
                "values": vec.values,
                "metadata": self._sanitize_metadata(vec.metadata)
            })
        
        if not pinecone_vectors:
            logger.error("No valid vectors to upsert after validation")
            return
        
        logger.info(f"Prepared {len(pinecone_vectors)} valid vectors for upsert")
        
        # Upsert in batches (Pinecone recommends 100 at a time)
        batch_size = 100
        total_upserted = 0
        for i in range(0, len(pinecone_vectors), batch_size):
            batch = pinecone_vectors[i:i + batch_size]
            try:
                logger.debug(f"Upserting batch {i//batch_size + 1} ({len(batch)} vectors)")
                self.index.upsert(vectors=batch, namespace=ns)
                total_upserted += len(batch)
            except Exception as e:
                logger.error(f"Failed to upsert batch {i//batch_size + 1}: {e}", exc_info=True)
                raise
        
        logger.info(f"Successfully upserted {total_upserted} vectors to Pinecone")
        
        # Verify upsert by checking index stats (with a small delay for eventual consistency)
        try:
            import time
            time.sleep(1)  # Brief delay for eventual consistency
            stats = self.index.describe_index_stats()
            ns_stats = stats.get('namespaces', {}).get(ns, {})
            ns_count = ns_stats.get('vector_count', 0)
            logger.info(f"Index stats after upsert - Namespace '{ns}': {ns_count} vectors")
        except Exception as e:
            logger.debug(f"Could not verify index stats after upsert: {e}")
    
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
    
    def get_index_stats(self) -> Dict[str, Any]:
        """Get current index statistics.
        
        Returns:
            Dict with index stats including total_vector_count and namespaces
        """
        try:
            stats = self.index.describe_index_stats()
            return {
                "total_vector_count": stats.get('total_vector_count', 0),
                "namespaces": stats.get('namespaces', {}),
                "dimension": stats.get('dimension', 0),
                "index_fullness": stats.get('index_fullness', 0),
            }
        except Exception as e:
            logger.error(f"Error getting index stats: {e}", exc_info=True)
            return {
                "total_vector_count": 0,
                "namespaces": {},
                "error": str(e),
            }
    
    def log_index_health(self) -> None:
        """Log index health information for monitoring."""
        stats = self.get_index_stats()
        logger.info(f"📊 Pinecone Index Health Report for '{self.config.index_name}':")
        logger.info(f"   Total vectors: {stats.get('total_vector_count', 0)}")
        logger.info(f"   Dimension: {stats.get('dimension', 0)}")
        logger.info(f"   Index fullness: {stats.get('index_fullness', 0):.2%}")
        
        namespaces = stats.get('namespaces', {})
        if namespaces:
            logger.info(f"   Namespaces ({len(namespaces)}):")
            for ns_name, ns_stats in namespaces.items():
                ns_count = ns_stats.get('vector_count', 0)
                logger.info(f"     - '{ns_name}': {ns_count} vectors")
        else:
            logger.warning("   ⚠️  No namespaces found - index may be empty or inactive")

