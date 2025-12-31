"""Common dataclasses for the RAG system."""

from dataclasses import dataclass
from typing import Any, Dict, List


@dataclass
class VectorRecord:
    """Vector record for upserting to vector store."""
    id: str
    values: List[float]
    metadata: Dict[str, Any]


@dataclass
class Candidate:
    """Candidate chunk from vector store query."""
    chunk_id: str
    score: float
    metadata: Dict[str, Any]


@dataclass
class CandidateText:
    """Candidate with text snippet for reranking."""
    chunk_id: str
    text_snippet: str
    metadata: Dict[str, Any]


@dataclass
class RerankResult:
    """Result from LLM reranking."""
    chunk_id: str
    relevance_score: int  # 0-100
    answerability: str  # "yes" or "no"
    key_evidence: List[str]


@dataclass
class ContextPack:
    """Final assembled context pack."""
    query: str
    selected_chunks: List[Dict[str, Any]]
    citations: List[Dict[str, Any]]
    trace: Dict[str, Any]

