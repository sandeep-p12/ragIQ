"""High-level pipeline functions for retrieval subsystem."""

import logging
import time
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

from src.config.retrieval import RetrievalConfig
from src.config.parsing import ParseForgeConfig
from src.core.dataclasses import CandidateText, ContextPack, VectorRecord
from src.core.interfaces import ContextAssembler
from src.providers.embedding.openai_embedding import OpenAIEmbeddingProvider
from src.storage.chunk.local import LocalChunkStore
from src.storage.vector.pinecone import PineconeVectorStore
from src.pipelines.retrieval.context_assembler import DefaultContextAssembler
from src.pipelines.retrieval.reranker import OpenAIReranker
from src.utils.io import load_jsonl


def ingest_from_chunking_outputs(
    children_jsonl_path: str,
    parents_jsonl_path: str,
    doc_id: str,
    cfg: RetrievalConfig = None
) -> Dict[str, Any]:
    """Ingest chunks from chunking output JSONL files.
    
    Args:
        children_jsonl_path: Path to children.jsonl file
        parents_jsonl_path: Path to parents.jsonl file
        doc_id: Document ID
        cfg: RetrievalConfig (defaults to RetrievalConfig.from_env())
        
    Returns:
        Dict with ingestion stats: {children_count, parents_count, avg_tokens, embedding_time, upsert_time, errors}
    """
    if cfg is None:
        cfg = RetrievalConfig.from_env()
    
    stats = {
        "children_count": 0,
        "parents_count": 0,
        "avg_tokens": 0,
        "embedding_time": 0,
        "upsert_time": 0,
        "errors": []
    }
    
    try:
        # Load JSONL files
        logger.info(f"Loading chunks from {children_jsonl_path} and {parents_jsonl_path}")
        children = load_jsonl(children_jsonl_path)
        parents = load_jsonl(parents_jsonl_path)
        
        stats["children_count"] = len(children)
        stats["parents_count"] = len(parents)
        
        logger.info(f"Loaded {len(children)} children and {len(parents)} parents for doc_id: {doc_id}")
        
        if not children:
            logger.warning(f"No children chunks to index for doc_id: {doc_id}")
            return stats
        
        # Calculate avg tokens
        token_counts = [c.get("token_count", 0) for c in children]
        stats["avg_tokens"] = sum(token_counts) / len(token_counts) if token_counts else 0
        
        # Store chunks locally
        logger.info(f"Storing {len(children)} children and {len(parents)} parents in LocalChunkStore")
        chunk_store = LocalChunkStore()
        chunk_store.put_chunks(children, parents)
        logger.info("Chunks stored locally successfully")
        
        # Embed and upsert children
        logger.info(f"Initializing embedding provider and vector store for doc_id: {doc_id}")
        embedding_provider = OpenAIEmbeddingProvider(cfg.embedding_config)
        vector_store = PineconeVectorStore(cfg.pinecone_config)
        logger.info(f"Vector store initialized. Index: {cfg.pinecone_config.index_name}, Namespace: {cfg.pinecone_config.namespace}")
        
        # Prepare children for embedding
        start_time = time.time()
        children_texts = [c.get("text_for_embedding", "") for c in children]
        children_embeddings = embedding_provider.embed_texts(children_texts)
        stats["embedding_time"] = time.time() - start_time
        
        # Prepare vector records for children
        children_vectors = []
        for chunk, embedding in zip(children, children_embeddings):
            metadata = {
                "doc_id": chunk.get("doc_id", doc_id),
                "parent_id": chunk.get("parent_id"),
                "page_span_start": chunk.get("page_span", [0, 0])[0] if isinstance(chunk.get("page_span"), list) else chunk.get("page_span", (0, 0))[0],
                "page_span_end": chunk.get("page_span", [0, 0])[1] if isinstance(chunk.get("page_span"), list) else chunk.get("page_span", (0, 0))[1],
                "section_label": chunk.get("section_label", ""),
                "header_path": chunk.get("header_path"),
                "element_type": chunk.get("element_type", ""),
                "token_count": chunk.get("token_count", 0),
                "structure_confidence": chunk.get("metadata", {}).get("structure_confidence", 1.0)
            }
            
            children_vectors.append(VectorRecord(
                id=chunk.get("chunk_id"),
                values=embedding,
                metadata=metadata
            ))
        
        # Upsert children
        start_time = time.time()
        try:
            logger.info(f"Upserting {len(children_vectors)} children vectors to Pinecone (namespace: children)")
            vector_store.upsert(children_vectors, namespace="children")
            stats["upsert_time"] = time.time() - start_time
            logger.info(f"Successfully upserted {len(children_vectors)} children vectors in {stats['upsert_time']:.2f}s")
        except Exception as e:
            stats["upsert_time"] = time.time() - start_time
            error_msg = f"Failed to upsert children vectors: {str(e)}"
            logger.error(error_msg, exc_info=True)
            stats["errors"].append(error_msg)
            raise  # Re-raise to ensure error is visible
        
        # Optionally embed and upsert parents
        if parents:
            try:
                parents_texts = [p.get("text_for_embedding", "") for p in parents]
                parents_embeddings = embedding_provider.embed_texts(parents_texts)
                
                parents_vectors = []
                for parent, embedding in zip(parents, parents_embeddings):
                    metadata = {
                        "doc_id": parent.get("doc_id", doc_id),
                        "parent_id": None,  # Parents don't have parents
                        "page_span_start": parent.get("page_span", [0, 0])[0] if isinstance(parent.get("page_span"), list) else parent.get("page_span", (0, 0))[0],
                        "page_span_end": parent.get("page_span", [0, 0])[1] if isinstance(parent.get("page_span"), list) else parent.get("page_span", (0, 0))[1],
                        "section_label": parent.get("section_label", ""),
                        "header_path": parent.get("header_path"),
                        "element_type": "parent",
                        "token_count": parent.get("token_count", 0),
                        "structure_confidence": parent.get("metadata", {}).get("structure_confidence", 1.0)
                    }
                    
                    parents_vectors.append(VectorRecord(
                        id=parent.get("chunk_id"),
                        values=embedding,
                        metadata=metadata
                    ))
                
                logger.info(f"Upserting {len(parents_vectors)} parent vectors to Pinecone (namespace: parents)")
                vector_store.upsert(parents_vectors, namespace="parents")
                logger.info(f"Successfully upserted {len(parents_vectors)} parent vectors")
            except Exception as e:
                error_msg = f"Failed to index parents: {str(e)}"
                logger.error(error_msg, exc_info=True)
                stats["errors"].append(error_msg)
        
        # Log index health after ingestion to monitor record counts
        try:
            vector_store.log_index_health()
        except Exception as e:
            logger.debug(f"Could not log index health: {e}")
    
    except Exception as e:
        stats["errors"].append(f"Ingestion failed: {str(e)}")
    
    return stats


def retrieve(
    query: str,
    filters: Optional[Dict[str, Any]] = None,
    cfg: RetrievalConfig = None,
    all_chunks: Optional[List[Dict[str, Any]]] = None,
    llm_config: Optional[ParseForgeConfig] = None
) -> ContextPack:
    """Retrieve and assemble context for a query.
    
    Args:
        query: Query string
        filters: Optional filters (e.g., {"doc_id": "doc1", "element_type": "table"})
        cfg: RetrievalConfig (defaults to RetrievalConfig.from_env())
        all_chunks: Optional list of all chunks for neighbor expansion (if None, will be loaded from store)
        llm_config: Optional ParseForgeConfig for LLM provider settings (for reranker).
                   If None, will be created from environment.
        
    Returns:
        ContextPack with selected chunks, citations, and trace
    """
    if cfg is None:
        cfg = RetrievalConfig.from_env()
    
    if filters is None:
        filters = {}
    
    # Initialize LLM config if not provided
    if llm_config is None:
        llm_config = ParseForgeConfig()
    
    # Initialize components
    embedding_provider = OpenAIEmbeddingProvider(cfg.embedding_config)
    vector_store = PineconeVectorStore(cfg.pinecone_config)
    reranker = OpenAIReranker(cfg.rerank_config, llm_config=llm_config)
    chunk_store = LocalChunkStore()
    context_assembler = DefaultContextAssembler(chunk_store)
    
    # Embed query
    query_embedding = embedding_provider.embed_query(query)
    
    # Query Pinecone
    top_k = cfg.pinecone_config.top_k_dense
    candidates = vector_store.query(
        vector=query_embedding,
        top_k=top_k,
        filters=filters,
        namespace="children"
    )
    
    # Store top 20 candidates in trace
    trace_candidates = [
        {
            "chunk_id": c.chunk_id,
            "score": c.score,
            "doc_id": c.metadata.get("doc_id"),
            "page_span": (c.metadata.get("page_span_start", 0), c.metadata.get("page_span_end", 0)),
            "section_label": c.metadata.get("section_label", ""),
            "element_type": c.metadata.get("element_type", "")
        }
        for c in candidates[:20]
    ]
    
    # Get doc_id from filters or first candidate
    doc_id = filters.get("doc_id")
    if not doc_id and candidates:
        doc_id = candidates[0].metadata.get("doc_id")
    
    if not doc_id:
        raise ValueError("doc_id is required. Provide via filters or ensure candidates have doc_id in metadata.")
    
    # Fetch candidate text snippets from chunk store
    candidate_metadata = {}
    candidate_texts = []
    
    for candidate in candidates[:cfg.rerank_config.max_candidates_to_rerank]:
        chunk_dict = chunk_store.get_chunk(doc_id, candidate.chunk_id, is_parent=False)
        if chunk_dict:
            text_snippet = chunk_dict.get("text_for_embedding", "")
            # Trim to max_text_chars_per_candidate
            if len(text_snippet) > cfg.rerank_config.max_text_chars_per_candidate:
                text_snippet = text_snippet[:cfg.rerank_config.max_text_chars_per_candidate]
            
            candidate_texts.append(CandidateText(
                chunk_id=candidate.chunk_id,
                text_snippet=text_snippet,
                metadata={
                    "doc_id": chunk_dict.get("doc_id"),
                    "section_label": chunk_dict.get("section_label", ""),
                    "header_path": chunk_dict.get("header_path"),
                    "page_span": chunk_dict.get("page_span"),
                    "element_type": chunk_dict.get("element_type", "")
                }
            ))
            
            candidate_metadata[candidate.chunk_id] = {
                "doc_id": chunk_dict.get("doc_id"),
                "section_label": chunk_dict.get("section_label", ""),
                "header_path": chunk_dict.get("header_path"),
                "page_span": chunk_dict.get("page_span"),
                "element_type": chunk_dict.get("element_type", "")
            }
    
    # Rerank candidates
    reranked = reranker.rerank(query, candidate_texts)
    
    # Load all chunks for neighbor expansion if not provided
    if all_chunks is None:
        # Try to load all children chunks for the doc from chunk store
        # This is a limitation - we'd need to know all chunk_ids or have an index
        # For now, we'll work with what we have (candidates + fetched chunks)
        # In production, we'd have a more efficient way to get all chunks
        all_chunks = []
        # We could scan the chunk store directory, but that's inefficient
        # For now, we'll use the chunks we've already fetched
        pass
    
    # Assemble context
    context_pack = context_assembler.assemble(
        query=query,
        reranked=reranked,
        cfg=cfg,
        doc_id=doc_id,
        candidate_metadata=candidate_metadata,
        all_chunks=all_chunks
    )
    
    # Add pinecone candidates to trace
    context_pack.trace["pinecone_candidates"] = trace_candidates
    
    return context_pack

