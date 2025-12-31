"""Unified orchestrator for the complete RAG pipeline.

This orchestrator coordinates parsing, chunking, indexing, and retrieval
into a single end-to-end workflow.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from src.config.chunking import ChunkConfig
from src.config.parsing import ParseForgeConfig
from src.config.parsing_strategies import StrategyEnum
from src.config.retrieval import RetrievalConfig
from src.pipelines.chunking.chunking import process_document
from src.pipelines.parsing.parseforge import ParseForge
from src.pipelines.retrieval.retrieval import ingest_from_chunking_outputs, retrieve
from src.schema.chunk import Chunk, ParentChunk
from src.schema.document import Document
from src.utils.io import load_jsonl, save_jsonl

logger = logging.getLogger(__name__)


class RAGOrchestrator:
    """Unified orchestrator for complete RAG pipeline."""
    
    def __init__(
        self,
        parsing_config: Optional[ParseForgeConfig] = None,
        chunking_config: Optional[ChunkConfig] = None,
        retrieval_config: Optional[RetrievalConfig] = None,
        progress_callback: Optional[callable] = None,
    ):
        """Initialize RAG orchestrator.
        
        Args:
            parsing_config: ParseForge configuration (defaults to ParseForgeConfig())
            chunking_config: Chunking configuration (defaults to ChunkConfig())
            retrieval_config: Retrieval configuration (defaults to RetrievalConfig.from_env())
            progress_callback: Optional callback for progress updates
        """
        self.parsing_config = parsing_config or ParseForgeConfig()
        self.chunking_config = chunking_config or ChunkConfig()
        self.retrieval_config = retrieval_config or RetrievalConfig.from_env()
        self.progress_callback = progress_callback
        
        # Initialize parsers
        self.parser = ParseForge(self.parsing_config, progress_callback=progress_callback)
    
    def parse(
        self,
        file_path: str,
        strategy: StrategyEnum = StrategyEnum.AUTO,
        start_page: Optional[int] = None,
        end_page: Optional[int] = None,
    ) -> Document:
        """Parse a document.
        
        Args:
            file_path: Path to document file
            strategy: Parsing strategy
            start_page: Optional start page (1-indexed)
            end_page: Optional end page (1-indexed)
            
        Returns:
            Document object
        """
        if self.progress_callback:
            self.progress_callback("parsing", 0.0, {"stage": "parse", "file": file_path})
        
        document = self.parser.parse(file_path, strategy, start_page, end_page)
        
        if self.progress_callback:
            self.progress_callback("parsing", 1.0, {"stage": "parse", "pages": len(document.pages)})
        
        return document
    
    def parse_to_markdown(
        self,
        file_path: str,
        strategy: StrategyEnum = StrategyEnum.AUTO,
        generate_image_descriptions: bool = True,
        start_page: Optional[int] = None,
        end_page: Optional[int] = None,
    ) -> str:
        """Parse document and convert to markdown.
        
        Args:
            file_path: Path to document file
            strategy: Parsing strategy
            generate_image_descriptions: Whether to generate image descriptions
            start_page: Optional start page (1-indexed)
            end_page: Optional end page (1-indexed)
            
        Returns:
            Markdown string
        """
        document = self.parse(file_path, strategy, start_page, end_page)
        return self.parser.to_markdown(document, generate_image_descriptions)
    
    def chunk(
        self,
        markdown_path: str,
        doc_id: str,
    ) -> Tuple[List[Chunk], List[ParentChunk], Dict[str, Any]]:
        """Chunk a markdown document.
        
        Args:
            markdown_path: Path to markdown file
            doc_id: Document ID
            
        Returns:
            Tuple of (children, parents, stats)
        """
        if self.progress_callback:
            self.progress_callback("chunking", 0.0, {"stage": "chunk", "file": markdown_path})
        
        children, parents, stats = process_document(
            markdown_path,
            self.chunking_config,
            doc_id
        )
        
        if self.progress_callback:
            self.progress_callback(
                "chunking",
                1.0,
                {
                    "stage": "chunk",
                    "children": len(children),
                    "parents": len(parents),
                    "stats": stats
                }
            )
        
        return children, parents, stats
    
    def index(
        self,
        children: List[Chunk],
        parents: List[ParentChunk],
        doc_id: str,
        output_dir: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Index chunks to vector store.
        
        Args:
            children: List of child chunks
            parents: List of parent chunks
            doc_id: Document ID
            output_dir: Optional directory to save JSONL files (defaults to temp)
            
        Returns:
            Dictionary with indexing stats
        """
        if self.progress_callback:
            self.progress_callback("indexing", 0.0, {"stage": "index", "doc_id": doc_id})
        
        # Save to JSONL files
        import tempfile
        import os
        
        if output_dir is None:
            output_dir = tempfile.mkdtemp()
        else:
            os.makedirs(output_dir, exist_ok=True)
        
        children_path = os.path.join(output_dir, f"{doc_id}_children.jsonl")
        parents_path = os.path.join(output_dir, f"{doc_id}_parents.jsonl")
        
        # Convert chunks to dictionaries
        children_dicts = [self._chunk_to_dict(chunk) for chunk in children]
        parents_dicts = [self._chunk_to_dict(chunk) for chunk in parents]
        
        save_jsonl(children_dicts, children_path)
        save_jsonl(parents_dicts, parents_path)
        
        # Ingest to vector store
        stats = ingest_from_chunking_outputs(
            children_path,
            parents_path,
            doc_id,
            self.retrieval_config
        )
        
        if self.progress_callback:
            self.progress_callback("indexing", 1.0, {"stage": "index", "stats": stats})
        
        return stats
    
    def query(
        self,
        query: str,
        filters: Optional[Dict[str, Any]] = None,
        all_chunks: Optional[List[Dict[str, Any]]] = None,
    ) -> Any:  # ContextPack
        """Query the indexed documents.
        
        Args:
            query: Query string
            filters: Optional metadata filters
            all_chunks: Optional list of all chunks (for retrieval demo)
            
        Returns:
            ContextPack with selected chunks, citations, and trace
        """
        if self.progress_callback:
            self.progress_callback("retrieval", 0.0, {"stage": "retrieve", "query": query})
        
        result = retrieve(
            query,
            filters or {},
            self.retrieval_config,
            all_chunks or []
        )
        
        if self.progress_callback:
            self.progress_callback("retrieval", 1.0, {"stage": "retrieve", "chunks": len(result.selected_chunks)})
        
        return result
    
    def pipeline(
        self,
        file_path: str,
        doc_id: str,
        strategy: StrategyEnum = StrategyEnum.AUTO,
        generate_image_descriptions: bool = True,
        index_chunks: bool = True,
    ) -> Dict[str, Any]:
        """Run complete pipeline: Parse → Chunk → Index.
        
        Args:
            file_path: Path to document file
            doc_id: Document ID
            strategy: Parsing strategy
            generate_image_descriptions: Whether to generate image descriptions
            index_chunks: Whether to index chunks to vector store
            
        Returns:
            Dictionary with pipeline results and stats
        """
        results = {
            "document": None,
            "markdown": None,
            "children": [],
            "parents": [],
            "chunking_stats": {},
            "indexing_stats": {},
        }
        
        # Step 1: Parse
        document = self.parse(file_path, strategy)
        results["document"] = document
        
        # Step 2: Convert to markdown
        markdown = self.parser.to_markdown(document, generate_image_descriptions)
        results["markdown"] = markdown
        
        # Step 3: Save markdown to temp file for chunking
        import tempfile
        with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as tmp_file:
            tmp_file.write(markdown)
            markdown_path = tmp_file.name
        
        # Step 4: Chunk
        children, parents, stats = self.chunk(markdown_path, doc_id)
        results["children"] = children
        results["parents"] = parents
        results["chunking_stats"] = stats
        
        # Step 5: Index (if requested)
        if index_chunks:
            indexing_stats = self.index(children, parents, doc_id)
            results["indexing_stats"] = indexing_stats
        
        return results
    
    def _chunk_to_dict(self, chunk: Chunk) -> Dict[str, Any]:
        """Convert chunk to dictionary for JSONL serialization."""
        return {
            "chunk_id": chunk.chunk_id,
            "doc_id": chunk.doc_id,
            "page_span": chunk.page_span,
            "page_nos": chunk.page_nos,
            "header_path": chunk.header_path,
            "section_label": chunk.section_label,
            "element_type": chunk.element_type,
            "raw_md_fragment": chunk.raw_md_fragment,
            "text_for_embedding": chunk.text_for_embedding,
            "metadata": chunk.metadata,
            "parent_id": chunk.parent_id,
            "token_count": chunk.token_count,
            "node_id": chunk.node_id,
            "line_start": chunk.line_start,
            "line_end": chunk.line_end,
        }

