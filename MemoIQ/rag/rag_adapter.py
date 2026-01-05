"""Thin wrappers around /src RAG functions - NO logic duplication."""

import json
import logging
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

from src.config.chunking import ChunkConfig
from src.config.parsing import ParseForgeConfig
from src.config.parsing_strategies import StrategyEnum
from src.config.retrieval import RetrievalConfig
from src.core.dataclasses import ContextPack
from src.pipelines.chunking.chunking import process_document
from src.pipelines.parsing.parseforge import ParseForge
from src.pipelines.retrieval.retrieval import ingest_from_chunking_outputs, retrieve
from src.schema.chunk import Chunk, ParentChunk


def parse_to_markdown(
    file_path: str,
    config: ParseForgeConfig,
    generate_image_descriptions: bool = True,
    strategy: Optional[StrategyEnum] = None,
) -> str:
    """
    Parse document to markdown using ParseForge.
    
    Args:
        file_path: Path to document file
        config: ParseForge configuration
        generate_image_descriptions: Whether to generate image descriptions
        strategy: Parsing strategy (if None, uses AUTO)
        
    Returns:
        Markdown string
    """
    parser = ParseForge(config)
    if strategy is None:
        strategy = StrategyEnum.AUTO
    
    # Log strategy with file name for clarity
    file_name = Path(file_path).name
    logger.info(f"Parsing {file_name} with strategy: {strategy}")
    
    # STRICT: Verify LLM_FULL bypasses all other parsing
    if strategy == StrategyEnum.LLM_FULL:
        logger.debug(f"LLM_FULL strategy: Will bypass layout detection, OCR, and table extraction")
    
    doc = parser.parse(file_path, strategy=strategy)
    return parser.to_markdown(doc, generate_image_descriptions=generate_image_descriptions)


def chunk_markdown(
    md_path: str,
    doc_id: str,
    config: ChunkConfig,
) -> Tuple[List[Chunk], List[ParentChunk], Dict[str, any]]:
    """
    Chunk markdown document using existing chunking pipeline.
    
    Args:
        md_path: Path to markdown file
        doc_id: Document ID
        config: ChunkConfig
        
    Returns:
        Tuple of (children, parents, stats)
    """
    return process_document(md_path, config, doc_id)


def index_chunks(
    children: List[Chunk],
    parents: List[ParentChunk],
    doc_id: str,
    config: RetrievalConfig,
) -> Dict[str, any]:
    """
    Index chunks to vector store using existing ingestion pipeline.
    
    Args:
        children: List of child chunks
        parents: List of parent chunks
        doc_id: Document ID
        config: RetrievalConfig
        
    Returns:
        Dict with indexing stats
    """
    # Save chunks to temporary JSONL files
    # Chunk and ParentChunk are dataclasses, so we can use asdict or __dict__
    from dataclasses import asdict
    
    logger.info(f"Serializing {len(children)} children and {len(parents)} parents to JSONL")
    
    # Validate chunks have required fields
    valid_children = []
    for chunk in children:
        if not hasattr(chunk, 'chunk_id') or not chunk.chunk_id:
            logger.warning(f"Skipping child chunk without chunk_id: {chunk}")
            continue
        if not hasattr(chunk, 'text_for_embedding') or not chunk.text_for_embedding:
            logger.warning(f"Child chunk {chunk.chunk_id} has no text_for_embedding")
        valid_children.append(chunk)
    
    if len(valid_children) < len(children):
        logger.warning(f"Filtered out {len(children) - len(valid_children)} invalid children chunks")
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as children_file:
        for chunk in valid_children:
            # Use asdict for dataclasses to ensure proper serialization
            chunk_dict = asdict(chunk)
            children_file.write(json.dumps(chunk_dict, default=str) + '\n')
        children_path = children_file.name
        logger.info(f"Saved {len(valid_children)} children to {children_path}")
    
    valid_parents = []
    for parent in parents:
        if not hasattr(parent, 'chunk_id') or not parent.chunk_id:
            logger.warning(f"Skipping parent chunk without chunk_id: {parent}")
            continue
        valid_parents.append(parent)
    
    if len(valid_parents) < len(parents):
        logger.warning(f"Filtered out {len(parents) - len(valid_parents)} invalid parent chunks")
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as parents_file:
        for parent in valid_parents:
            parent_dict = asdict(parent)
            parents_file.write(json.dumps(parent_dict, default=str) + '\n')
        parents_path = parents_file.name
        logger.info(f"Saved {len(valid_parents)} parents to {parents_path}")
    
    try:
        # Call existing ingestion function
        logger.info(f"Indexing {len(children)} children and {len(parents)} parents for doc_id: {doc_id}")
        stats = ingest_from_chunking_outputs(children_path, parents_path, doc_id, config)
        
        # Log indexing results
        if stats.get("errors"):
            logger.error(f"Indexing errors for {doc_id}: {stats['errors']}")
        else:
            logger.info(f"Successfully indexed {doc_id}: {stats.get('children_count', 0)} children, {stats.get('parents_count', 0)} parents")
            logger.info(f"  - Embedding time: {stats.get('embedding_time', 0):.2f}s")
            logger.info(f"  - Upsert time: {stats.get('upsert_time', 0):.2f}s")
        
        return stats
    except Exception as e:
        logger.error(f"Failed to index chunks for {doc_id}: {e}", exc_info=True)
        return {"error": str(e), "children_count": 0, "parents_count": 0}
    finally:
        # Clean up temp files
        Path(children_path).unlink(missing_ok=True)
        Path(parents_path).unlink(missing_ok=True)


def rag_retrieve(
    query: str,
    doc_id: Optional[str],
    config: RetrievalConfig,
    all_chunks: List[Dict[str, any]],
    llm_config: ParseForgeConfig,
) -> ContextPack:
    """
    Retrieve context using existing retrieval pipeline.
    
    Args:
        query: Query string
        doc_id: Document ID (None to search all documents)
        config: RetrievalConfig
        all_chunks: List of all chunks for neighbor expansion
        llm_config: ParseForgeConfig for LLM provider settings
        
    Returns:
        ContextPack with selected chunks, citations, and trace
    """
    # If doc_id is None, search across all documents (no filter)
    # If doc_id is provided, filter to that specific document
    filters = {}
    if doc_id and doc_id != "unknown":
        filters = {"doc_id": doc_id}
    
    logger.debug(f"RAG retrieve: query='{query[:50]}...', doc_id={doc_id}, filters={filters}")
    return retrieve(query, filters, config, all_chunks, llm_config=llm_config)

