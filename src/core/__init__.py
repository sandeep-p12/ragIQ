"""Core interfaces, protocols, and dataclasses."""

from src.core.dataclasses import (
    Candidate,
    CandidateText,
    ContextPack,
    RerankResult,
    VectorRecord,
)
from src.core.interfaces import (
    ChunkStore,
    ContextAssembler,
    EmbeddingProvider,
    Reranker,
    VectorStore,
)

__all__ = [
    "VectorRecord",
    "Candidate",
    "CandidateText",
    "RerankResult",
    "ContextPack",
    "EmbeddingProvider",
    "VectorStore",
    "ChunkStore",
    "Reranker",
    "ContextAssembler",
]

