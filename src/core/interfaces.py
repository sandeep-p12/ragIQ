"""Protocol-based interfaces for all providers and storage."""

from typing import Any, Dict, List, Optional, Protocol


class EmbeddingProvider(Protocol):
    """Protocol for embedding providers."""
    
    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Embed a batch of texts.
        
        Args:
            texts: List of text strings to embed
            
        Returns:
            List of embedding vectors (list of floats)
        """
        ...
    
    def embed_query(self, query: str) -> List[float]:
        """Embed a single query string.
        
        Args:
            query: Query string to embed
            
        Returns:
            Embedding vector (list of floats)
        """
        ...


class VectorStore(Protocol):
    """Protocol for vector stores."""
    
    def upsert(self, vectors: List["VectorRecord"], namespace: Optional[str] = None) -> None:
        """Upsert vectors to the store.
        
        Args:
            vectors: List of vector records to upsert
            namespace: Optional namespace (defaults to config namespace)
        """
        ...
    
    def query(
        self,
        vector: List[float],
        top_k: int,
        filters: Optional[Dict[str, Any]] = None,
        namespace: Optional[str] = None
    ) -> List["Candidate"]:
        """Query the vector store.
        
        Args:
            vector: Query embedding vector
            top_k: Number of results to return
            filters: Optional metadata filters
            namespace: Optional namespace (defaults to config namespace)
            
        Returns:
            List of candidate chunks
        """
        ...


class ChunkStore(Protocol):
    """Protocol for chunk storage."""
    
    def put_chunks(
        self,
        children: List[Dict[str, Any]],
        parents: List[Dict[str, Any]]
    ) -> None:
        """Store chunks and parents.
        
        Args:
            children: List of child chunk dictionaries
            parents: List of parent chunk dictionaries
        """
        ...
    
    def get_chunk(
        self,
        doc_id: str,
        chunk_id: str,
        is_parent: bool = False
    ) -> Optional[Dict[str, Any]]:
        """Get a single chunk.
        
        Args:
            doc_id: Document ID
            chunk_id: Chunk ID
            is_parent: Whether this is a parent chunk
            
        Returns:
            Chunk dictionary or None if not found
        """
        ...
    
    def get_chunks_bulk(
        self,
        keys: List[tuple[str, str, bool]]  # (doc_id, chunk_id, is_parent)
    ) -> List[Optional[Dict[str, Any]]]:
        """Get multiple chunks in bulk.
        
        Args:
            keys: List of (doc_id, chunk_id, is_parent) tuples
            
        Returns:
            List of chunk dictionaries (None for missing chunks)
        """
        ...


class Reranker(Protocol):
    """Protocol for rerankers."""
    
    def rerank(
        self,
        query: str,
        candidates: List["CandidateText"]
    ) -> List["RerankResult"]:
        """Rerank candidates using LLM.
        
        Args:
            query: Query string
            candidates: List of candidate texts to rerank
            
        Returns:
            List of reranked results sorted by relevance_score (descending)
        """
        ...


class ContextAssembler(Protocol):
    """Protocol for context assemblers."""
    
    def assemble(
        self,
        query: str,
        reranked: List["RerankResult"],
        cfg: Any  # RetrievalConfig
    ) -> "ContextPack":
        """Assemble final context pack from reranked results.
        
        Args:
            query: Original query string
            reranked: List of reranked results
            cfg: RetrievalConfig with assembly parameters
            
        Returns:
            ContextPack with selected chunks, citations, and trace
        """
        ...


# Additional protocols for parsing providers
class LLMProvider(Protocol):
    """Protocol for LLM providers."""
    
    def generate(
        self,
        prompt: str,
        model: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: Optional[int] = None
    ) -> str:
        """Generate text from prompt.
        
        Args:
            prompt: Input prompt
            model: Model name (optional, uses default if not provided)
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            
        Returns:
            Generated text
        """
        ...


class OCRProvider(Protocol):
    """Protocol for OCR providers."""
    
    def detect_text(
        self,
        image: Any,  # PIL Image or numpy array
        page_index: int
    ) -> "TextDetection":
        """Detect and recognize text in image.
        
        Args:
            image: Image to process
            page_index: Page index for metadata
            
        Returns:
            TextDetection result
        """
        ...


class LayoutProvider(Protocol):
    """Protocol for layout detection providers."""
    
    def detect_layout(
        self,
        image: Any,  # PIL Image or numpy array
        page_index: int
    ) -> List["LayoutDetectionOutput"]:
        """Detect layout elements in image.
        
        Args:
            image: Image to process
            page_index: Page index for metadata
            
        Returns:
            List of layout detection outputs
        """
        ...

