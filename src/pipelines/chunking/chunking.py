"""Main orchestration pipeline: LlamaIndex-first hybrid chunking."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple

from src.config.chunking import ChunkConfig
from src.pipelines.chunking.chunkers import Chunk, StructureFirstChunker, TokenBudgetRefiner
from src.pipelines.chunking.element_extractor import build_header_path_stack, extract_elements_from_nodes
from src.pipelines.chunking.hierarchy import (
    ParentChunk,
    assign_parent_ids,
    create_parents_hybrid,
    generate_stable_ids,
)
from src.pipelines.chunking.llama_parser import parse_with_llamaindex
from src.pipelines.chunking.page_parser import merge_continuations, split_into_page_blocks
from src.pipelines.chunking.repair import apply_repair_mode


def process_document(
    file_path: str,
    config: ChunkConfig,
    doc_id: str
) -> Tuple[List[Chunk], List[ParentChunk], Dict[str, Any]]:
    """Process document through complete chunking pipeline.
    
    Pipeline flow:
    1. Read Markdown file
    2. Apply repair mode → RepairResult
    3. Normalize page markers
    4. Split into PageBlocks with repair metadata
    5. Merge continuations (if enabled and high confidence)
    6. For each PageBlock:
       - Parse with LlamaIndex (MarkdownNodeParser → HierarchicalNodeParser)
       - Extract typed elements
       - Generate section_labels
    7. Structure-first chunking (create candidate chunks)
    8. Token-budget refinement (split oversized, merge tiny)
    9. Hybrid hierarchy building (parents + children)
    10. Typed serialization (raw_md_fragment + text_for_embedding)
    11. Generate stable IDs
    12. Return (children, parents, stats)
    
    Args:
        file_path: Path to Markdown file
        config: ChunkConfig with all tunables
        doc_id: Document ID
    
    Returns:
        Tuple[List[Chunk], List[ParentChunk], Dict[str, Any]]: 
        (children, parents, processing_stats)
    """
    # 1. Read Markdown file
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()
    
    # 2. Apply repair mode
    repair_result = apply_repair_mode(content)
    
    # 3-4. Split into PageBlocks
    page_blocks = split_into_page_blocks(repair_result.repaired_content, repair_result)
    
    # 5. Merge continuations
    if config.enable_cross_page_merge:
        page_blocks = merge_continuations(page_blocks, config)
    
    # 6. Parse with LlamaIndex and extract elements
    # Track elements per page block
    page_block_elements = []  # List of (page_block, elements, header_paths) tuples
    all_elements = []  # Collect all elements for stats
    
    for block_idx, page_block in enumerate(page_blocks):
        # Parse with LlamaIndex
        nodes = parse_with_llamaindex(page_block, use_hierarchical=True)
        
        # Extract elements
        block_elements = extract_elements_from_nodes(nodes)
        all_elements.extend(block_elements)  # Collect for stats
        
        # Build header paths for this block FIRST
        # This ensures elements can inherit nearest preceding headings
        block_header_paths = build_header_path_stack(block_elements)
        
        # Generate section labels with header paths (for metadata storage)
        # Note: section labels will be regenerated in chunkers.py with proper header_paths
        for elem_idx, element in enumerate(block_elements):
            header_path = block_header_paths.get(elem_idx)
            from src.pipelines.chunking.element_extractor import generate_section_label
            section_label = generate_section_label(
                element,
                header_path,  # Now we have header_paths
                page_block.page_no,
                elem_idx  # Use element index, not block index
            )
            # Store section label in element metadata (if element supports it)
            if hasattr(element, 'section_label'):
                element.section_label = section_label
        
        page_block_elements.append((page_block, block_elements, block_header_paths))
    
    # 7. Structure-first chunking
    structure_chunker = StructureFirstChunker(config)
    all_chunks = []
    
    # Process each page block's elements
    for page_block, block_elements, block_header_paths in page_block_elements:
        # Get page span
        page_span = page_block.page_span
        page_nos = page_block.page_nos
        
        # Create candidate chunks for this block
        candidate_chunks = structure_chunker.chunk(
            block_elements,
            page_span,
            page_nos,
            doc_id,
            block_header_paths
        )
        
        # Add page metadata to chunks and adjust line positions to absolute
        for chunk in candidate_chunks:
            chunk.metadata["structure_confidence"] = page_block.structure_confidence
            if page_block.repair_applied:
                chunk.metadata["repair_applied"] = [
                    {
                        "repair_type": r.repair_type,
                        "location": r.location,
                        "reason": r.reason
                    }
                    for r in page_block.repair_applied
                ]
            
            # Adjust line positions to absolute (add page_block.start_line offset)
            # Note: element line_start/line_end are relative to node text, not page block
            # Adding page_block.start_line gives us an approximation of absolute position
            # A more accurate approach would require tracking element positions within page block
            if chunk.line_start is not None:
                chunk.line_start = page_block.start_line + chunk.line_start
            if chunk.line_end is not None:
                chunk.line_end = page_block.start_line + chunk.line_end
        
        all_chunks.extend(candidate_chunks)
    
    # 8. Token-budget refinement
    refiner = TokenBudgetRefiner(config)
    refined_chunks = refiner.refine(all_chunks)
    
    # 9. Hybrid hierarchy building
    parents = create_parents_hybrid(refined_chunks, config)
    children = assign_parent_ids(refined_chunks, parents)
    
    # 10. Typed serialization (already done in chunk creation)
    # Ensure all chunks have both raw_md_fragment and text_for_embedding
    
    # 11. Generate stable IDs
    children = generate_stable_ids(children)
    parents = generate_stable_ids(parents)  # Parents are also chunks
    
    # 12. Compute processing stats
    stats = _compute_processing_stats(
        page_blocks, all_elements, children, parents, repair_result
    )
    
    return children, parents, stats


def _compute_processing_stats(
    page_blocks: List,
    elements: List,
    children: List[Chunk],
    parents: List[ParentChunk],
    repair_result
) -> Dict[str, Any]:
    """Compute processing statistics.
    
    Args:
        page_blocks: List of page blocks
        elements: List of extracted elements
        children: List of child chunks
        parents: List of parent chunks
        repair_result: Repair result
    
    Returns:
        Dict[str, Any]: Processing statistics
    """
    # Count elements by type
    elements_by_type = {}
    for element in elements:
        elem_type = type(element).__name__.lower()
        elements_by_type[elem_type] = elements_by_type.get(elem_type, 0) + 1
    
    # Count chunks by type
    chunks_by_type = {}
    for chunk in children:
        chunk_type = chunk.element_type
        chunks_by_type[chunk_type] = chunks_by_type.get(chunk_type, 0) + 1
    
    # Count merges (chunks with merged_from metadata)
    merges_performed = sum(
        1 for chunk in children
        if "merged_from" in chunk.metadata
    )
    
    # Compute token statistics
    token_counts = [chunk.token_count for chunk in children]
    avg_tokens = sum(token_counts) / len(token_counts) if token_counts else 0
    max_tokens = max(token_counts) if token_counts else 0
    
    # Repair summary
    repair_summary = {
        "total_repairs": sum(
            len(records) for records in repair_result.repair_applied.values()
        ),
        "by_type": {
            repair_type: len(records)
            for repair_type, records in repair_result.repair_applied.items()
        }
    }
    
    # Structure confidence average
    structure_confidence_avg = repair_result.structure_confidence
    
    return {
        "pages_processed": len(page_blocks),
        "elements_by_type": elements_by_type,
        "chunks_by_type": chunks_by_type,
        "merges_performed": merges_performed,
        "avg_token_size": avg_tokens,
        "max_token_size": max_tokens,
        "repair_applied_summary": repair_summary,
        "structure_confidence_avg": structure_confidence_avg,
        "total_children": len(children),
        "total_parents": len(parents),
    }

