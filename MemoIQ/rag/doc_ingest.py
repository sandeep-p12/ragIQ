"""Document ingestion orchestration."""

import logging
from pathlib import Path
from typing import Dict, List, Tuple

from src.config.chunking import ChunkConfig
from src.config.parsing import ParseForgeConfig
from src.config.parsing_strategies import StrategyEnum
from src.config.retrieval import RetrievalConfig
from src.schema.chunk import Chunk, ParentChunk

from MemoIQ.rag.rag_adapter import (
    chunk_markdown,
    index_chunks,
    parse_to_markdown,
)

logger = logging.getLogger(__name__)


def ingest_reference_documents(
    document_paths: List[str],
    parsing_config: ParseForgeConfig,
    chunking_config: ChunkConfig,
    retrieval_config: RetrievalConfig,
    index_documents: bool = True,
    strategy: StrategyEnum = StrategyEnum.AUTO,
) -> Dict[str, Tuple[List[Chunk], List[ParentChunk], Dict[str, any]]]:
    """
    Ingest reference documents through parse → chunk → index workflow.
    
    Args:
        document_paths: List of paths to reference documents
        parsing_config: ParseForge configuration
        chunking_config: ChunkConfig
        retrieval_config: RetrievalConfig
        index_documents: Whether to index chunks to vector store
        strategy: Parsing strategy to use
        
    Returns:
        Dict mapping doc_id -> (children, parents, stats)
    """
    results = {}
    
    for doc_path in document_paths:
        doc_path_obj = Path(doc_path)
        doc_id = doc_path_obj.stem  # Use filename without extension as doc_id
        
        logger.info(f"Ingesting document: {doc_path} (doc_id: {doc_id})")
        
        try:
            # Step 1: Parse to markdown
            logger.info(f"Parsing {doc_id}...")
            markdown = parse_to_markdown(doc_path, parsing_config, strategy=strategy)
            
            # Save markdown to temp file for chunking
            import tempfile
            with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False, encoding='utf-8') as tmp_md:
                tmp_md.write(markdown)
                tmp_md_path = tmp_md.name
            
            try:
                # Step 2: Chunk markdown
                logger.info(f"Chunking {doc_id}...")
                children, parents, chunk_stats = chunk_markdown(tmp_md_path, doc_id, chunking_config)
                
                # Step 3: Index chunks (if requested)
                index_stats = {}
                if index_documents:
                    logger.info(f"Indexing {doc_id}...")
                    index_stats = index_chunks(children, parents, doc_id, retrieval_config)
                
                # Combine stats
                stats = {**chunk_stats, "indexing": index_stats}
                
                results[doc_id] = (children, parents, stats)
                logger.info(f"Successfully ingested {doc_id}: {len(children)} children, {len(parents)} parents")
                
            finally:
                # Clean up temp markdown file
                Path(tmp_md_path).unlink(missing_ok=True)
                
        except Exception as e:
            logger.error(f"Error ingesting {doc_path}: {e}", exc_info=True)
            # Continue with other documents
            results[doc_id] = ([], [], {"error": str(e)})
    
    return results

