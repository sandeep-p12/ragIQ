"""Context assembler with hierarchy-aware expansion and token budgeting."""

from typing import Any, Dict, List

from src.pipelines.chunking.chunkers import Chunk
from src.core.dataclasses import ContextPack, RerankResult
from src.core.interfaces import ChunkStore, ContextAssembler
from src.pipelines.chunking.retrieval_safety import expand_neighbors
from src.utils.tokens import count_tokens


class DefaultContextAssembler(ContextAssembler):
    """Assembles final context using hierarchy-aware expansion and token budgeting."""
    
    def __init__(self, chunk_store: ChunkStore):
        """Initialize context assembler.
        
        Args:
            chunk_store: ChunkStore for fetching chunks
        """
        self.chunk_store = chunk_store
    
    def assemble(
        self,
        query: str,
        reranked: List[RerankResult],
        cfg: Any,  # RetrievalConfig
        doc_id: str = None,
        candidate_metadata: Dict[str, Dict[str, Any]] = None,
        all_chunks: List[Dict[str, Any]] = None
    ) -> ContextPack:
        """Assemble final context pack from reranked results.
        
        Args:
            query: Original query string
            reranked: List of reranked results (sorted by relevance_score descending)
            cfg: RetrievalConfig with assembly parameters
            doc_id: Document ID (required for fetching chunks)
            candidate_metadata: Dict mapping chunk_id to metadata (includes doc_id)
            
        Returns:
            ContextPack with selected chunks, citations, and trace
        """
        trace = {
            "pinecone_candidates": [],  # Will be populated by pipeline
            "rerank_results": [{"chunk_id": r.chunk_id, "relevance_score": r.relevance_score} for r in reranked],
            "expansion_added": [],
            "dropped_ids": [],
            "final_token_count": 0
        }
        
        # A) Start with reranked top results (top return_top_n)
        top_n = min(cfg.rerank_config.return_top_n, len(reranked))
        primary_chunk_ids = [r.chunk_id for r in reranked[:top_n]]
        
        # B) Fetch primary chunks from ChunkStore
        # Get doc_id from candidate_metadata or use provided doc_id
        if candidate_metadata:
            # Extract doc_ids from metadata
            doc_ids = set()
            for chunk_id in primary_chunk_ids:
                if chunk_id in candidate_metadata:
                    meta = candidate_metadata[chunk_id]
                    if "doc_id" in meta:
                        doc_ids.add(meta["doc_id"])
            
            if doc_ids:
                doc_id = list(doc_ids)[0]  # Use first doc_id found
        
        if not doc_id:
            raise ValueError("doc_id is required for context assembly. Provide via parameter or candidate_metadata.")
        
        # Fetch primary chunks
        keys = [(doc_id, chunk_id, False) for chunk_id in primary_chunk_ids]
        primary_chunk_dicts = self.chunk_store.get_chunks_bulk(keys)
        primary_chunk_dicts = [c for c in primary_chunk_dicts if c is not None]
        
        # Get all chunks for neighbor expansion
        if all_chunks is None:
            # If not provided, we'll work with just primary chunks (limited neighbor expansion)
            all_chunk_dicts = primary_chunk_dicts.copy()
        else:
            # Use provided all_chunks
            all_chunk_dicts = all_chunks
        
        # Convert dicts to Chunk objects for expand_neighbors
        all_chunks_list = self._dicts_to_chunks(all_chunk_dicts)
        primary_chunks_list = [c for c in all_chunks_list if c.chunk_id in primary_chunk_ids]
        
        # C) Include parent chunks
        parent_chunks_list = []
        if cfg.include_parents:
            for chunk in primary_chunks_list:
                if chunk.parent_id:
                    parent_dict = self._fetch_parent_chunk(chunk.parent_id, chunk.doc_id)
                    if parent_dict:
                        parent_chunk = self._dict_to_chunk(parent_dict, is_parent=True)
                        parent_chunks_list.append(parent_chunk)
                        trace["expansion_added"].append(f"parent:{chunk.parent_id}")
        
        # D) Expand neighbors using retrieval_safety.expand_neighbors()
        # Create a mock config object for expand_neighbors
        class MockConfig:
            def __init__(self, neighbor_same_page, neighbor_cross_page):
                self.neighbor_same_page = neighbor_same_page
                self.neighbor_cross_page = neighbor_cross_page
        
        mock_config = MockConfig(cfg.neighbor_same_page, cfg.neighbor_cross_page)
        
        neighbor_chunks_list = []
        for chunk in primary_chunks_list:
            expanded = expand_neighbors(chunk, all_chunks_list, mock_config)
            # Remove the original chunk and primary chunks
            for neighbor in expanded[1:]:  # Skip first (original)
                if neighbor.chunk_id not in primary_chunk_ids and neighbor not in neighbor_chunks_list:
                    neighbor_chunks_list.append(neighbor)
                    trace["expansion_added"].append(f"neighbor:{neighbor.chunk_id}")
        
        # E) Ordering and deduplication
        # 1. Primary reranked child chunks (highest score first)
        # 2. Their parent chunks (immediately after each child)
        # 3. Same-parent neighbors (siblings)
        # 4. Cross-page neighbors (last)
        
        ordered_chunks = []
        seen_ids = set()
        
        # Add primary chunks in rerank order
        rerank_map = {r.chunk_id: r for r in reranked}
        for chunk in primary_chunks_list:
            if chunk.chunk_id not in seen_ids:
                ordered_chunks.append(("primary", chunk, rerank_map.get(chunk.chunk_id, None)))
                seen_ids.add(chunk.chunk_id)
                
                # Add parent immediately after
                parent = next((p for p in parent_chunks_list if p.chunk_id == chunk.parent_id), None)
                if parent and parent.chunk_id not in seen_ids:
                    ordered_chunks.append(("parent", parent, None))
                    seen_ids.add(parent.chunk_id)
        
        # Add neighbors (siblings first, then cross-page)
        # Separate siblings from cross-page
        sibling_neighbors = []
        cross_page_neighbors = []
        
        for neighbor in neighbor_chunks_list:
            if neighbor.chunk_id in seen_ids:
                continue
            
            # Check if it's a sibling (same parent as a primary chunk)
            is_sibling = any(
                neighbor.parent_id == p.parent_id and neighbor.parent_id
                for p in primary_chunks_list
            )
            
            if is_sibling:
                sibling_neighbors.append(neighbor)
            else:
                cross_page_neighbors.append(neighbor)
        
        # Add siblings
        for neighbor in sibling_neighbors:
            if neighbor.chunk_id not in seen_ids:
                ordered_chunks.append(("sibling", neighbor, None))
                seen_ids.add(neighbor.chunk_id)
        
        # Add cross-page neighbors
        for neighbor in cross_page_neighbors:
            if neighbor.chunk_id not in seen_ids:
                ordered_chunks.append(("cross_page", neighbor, None))
                seen_ids.add(neighbor.chunk_id)
        
        # F) Apply token budget
        selected_chunks = []
        total_tokens = 0
        min_keep = cfg.min_primary_hits_to_keep
        
        # Always keep top min_keep primary chunks
        primary_kept = 0
        for chunk_type, chunk, rerank_result in ordered_chunks:
            chunk_tokens = count_tokens(chunk.text_for_embedding)
            
            # Never drop top min_keep primary chunks
            if chunk_type == "primary" and primary_kept < min_keep:
                selected_chunks.append((chunk_type, chunk, rerank_result))
                total_tokens += chunk_tokens
                primary_kept += 1
                continue
            
            # Check if adding this chunk would exceed budget
            if total_tokens + chunk_tokens > cfg.final_max_tokens:
                # Drop in priority order: cross_page > sibling > parent > primary
                if chunk_type == "cross_page":
                    trace["dropped_ids"].append(chunk.chunk_id)
                    continue
                elif chunk_type == "sibling":
                    trace["dropped_ids"].append(chunk.chunk_id)
                    continue
                elif chunk_type == "parent":
                    trace["dropped_ids"].append(chunk.chunk_id)
                    continue
                elif chunk_type == "primary":
                    trace["dropped_ids"].append(chunk.chunk_id)
                    continue
            
            selected_chunks.append((chunk_type, chunk, rerank_result))
            total_tokens += chunk_tokens
        
        # G) Convert to final format
        final_chunk_dicts = []
        citations = []
        
        for chunk_type, chunk, rerank_result in selected_chunks:
            chunk_dict = self._chunk_to_dict(chunk)
            final_chunk_dicts.append(chunk_dict)
            
            citations.append({
                "doc_id": chunk.doc_id,
                "chunk_id": chunk.chunk_id,
                "page_span": chunk.page_span,
                "section_label": chunk.section_label
            })
        
        trace["final_token_count"] = total_tokens
        
        return ContextPack(
            query=query,
            selected_chunks=final_chunk_dicts,
            citations=citations,
            trace=trace
        )
    
    
    def _fetch_parent_chunk(self, parent_id: str, doc_id: str) -> Dict[str, Any]:
        """Fetch parent chunk.
        
        Args:
            parent_id: Parent chunk ID
            doc_id: Document ID
            
        Returns:
            Parent chunk dictionary or None
        """
        return self.chunk_store.get_chunk(doc_id, parent_id, is_parent=True)
    
    def _dict_to_chunk(self, chunk_dict: Dict[str, Any], is_parent: bool = False) -> Chunk:
        """Convert chunk dict to Chunk object.
        
        Args:
            chunk_dict: Chunk dictionary
            is_parent: Whether this is a parent chunk
            
        Returns:
            Chunk object
        """
        return Chunk(
            chunk_id=chunk_dict["chunk_id"],
            doc_id=chunk_dict["doc_id"],
            page_span=tuple(chunk_dict["page_span"]) if isinstance(chunk_dict["page_span"], list) else chunk_dict["page_span"],
            page_nos=chunk_dict["page_nos"],
            header_path=chunk_dict.get("header_path"),
            section_label=chunk_dict["section_label"],
            element_type=chunk_dict["element_type"],
            raw_md_fragment=chunk_dict["raw_md_fragment"],
            text_for_embedding=chunk_dict["text_for_embedding"],
            metadata=chunk_dict.get("metadata", {}),
            parent_id=chunk_dict.get("parent_id"),
            token_count=chunk_dict.get("token_count", 0),
            node_id=chunk_dict.get("node_id", ""),
            line_start=chunk_dict.get("line_start"),
            line_end=chunk_dict.get("line_end")
        )
    
    def _dicts_to_chunks(self, chunk_dicts: List[Dict[str, Any]]) -> List[Chunk]:
        """Convert list of dicts to chunks.
        
        Args:
            chunk_dicts: List of chunk dictionaries
            
        Returns:
            List of Chunk objects
        """
        return [self._dict_to_chunk(d) for d in chunk_dicts]
    
    def _chunk_to_dict(self, chunk: Chunk) -> Dict[str, Any]:
        """Convert Chunk object to dict.
        
        Args:
            chunk: Chunk object
            
        Returns:
            Chunk dictionary
        """
        result = {
            "chunk_id": chunk.chunk_id,
            "doc_id": chunk.doc_id,
            "page_span": list(chunk.page_span) if isinstance(chunk.page_span, tuple) else chunk.page_span,
            "page_nos": chunk.page_nos,
            "header_path": chunk.header_path,
            "section_label": chunk.section_label,
            "element_type": chunk.element_type,
            "raw_md_fragment": chunk.raw_md_fragment,
            "text_for_embedding": chunk.text_for_embedding,
            "metadata": chunk.metadata,
            "parent_id": chunk.parent_id,
            "token_count": chunk.token_count,
            "node_id": chunk.node_id
        }
        
        # Add line position information if available
        if hasattr(chunk, 'line_start') and chunk.line_start is not None:
            result["line_start"] = chunk.line_start
        if hasattr(chunk, 'line_end') and chunk.line_end is not None:
            result["line_end"] = chunk.line_end
        
        return result

