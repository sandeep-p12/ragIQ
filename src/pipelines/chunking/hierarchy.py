"""Hybrid parent grouping: heading-based or soft sections, heading-independent."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from src.pipelines.chunking.chunkers import Chunk
from src.schema.chunk import ParentChunk
from src.utils.ids import generate_chunk_id

# ParentChunk is now imported from schema


def create_parents_hybrid(
    chunks: List[Chunk],
    config
) -> List[ParentChunk]:
    """Create parent chunks using hybrid grouping strategy.
    
    If heading structure is reliable (structure_confidence > threshold):
    - Group by heading level (H2/H3 configurable)
    Else (heading structure unreliable or missing):
    - Group by soft sections within page windows
    - Keep element integrity (don't split elements across parents)
    - Use section_label as grouping key
    
    Args:
        chunks: List of child chunks
        config: ChunkConfig with grouping settings
    
    Returns:
        List[ParentChunk]: Parent chunks
    """
    if not chunks:
        return []
    
    # Determine grouping strategy based on structure confidence
    avg_confidence = _compute_avg_structure_confidence(chunks)
    use_heading_based = avg_confidence >= config.structure_confidence_threshold
    
    if use_heading_based:
        return _create_heading_based_parents(chunks, config)
    else:
        return _create_soft_section_parents(chunks, config)


def _compute_avg_structure_confidence(chunks: List[Chunk]) -> float:
    """Compute average structure confidence from chunks."""
    if not chunks:
        return 0.0
    
    confidences = []
    for chunk in chunks:
        conf = chunk.metadata.get("structure_confidence", 1.0)
        confidences.append(conf)
    
    return sum(confidences) / len(confidences) if confidences else 0.0


def _create_heading_based_parents(
    chunks: List[Chunk],
    config
) -> List[ParentChunk]:
    """Create parents based on heading hierarchy.
    
    Groups chunks by heading at target level. Chunks without headings
    are grouped with the preceding heading group, or together if consecutive.
    """
    parents = []
    current_parent_chunks = []
    current_header_path = None
    target_level = config.parent_heading_level
    
    for chunk in chunks:
        # Check if chunk has a heading at target level
        header_path = chunk.header_path
        if header_path:
            # Extract heading level from path
            # Path format: "H1 > H2 > H3"
            path_parts = header_path.split(" > ")
            
            # Find heading at target level
            target_heading = None
            if len(path_parts) >= target_level:
                target_heading = path_parts[target_level - 1]
            
            # Check if we should start a new parent
            if target_heading and target_heading != current_header_path:
                # Create parent from previous group
                if current_parent_chunks:
                    parent = _create_parent_from_chunks(
                        current_parent_chunks,
                        current_header_path or "Root",
                        "heading_based"
                    )
                    parents.append(parent)
                
                # Start new group
                current_header_path = target_heading
                current_parent_chunks = [chunk]
            else:
                # Same heading or no target heading found - add to current group
                current_parent_chunks.append(chunk)
        else:
            # No header path - add to current group if it exists
            # This ensures chunks without headings are grouped with preceding chunks
            # that have headings, avoiding 1:1 relationships
            if current_parent_chunks:
                current_parent_chunks.append(chunk)
            else:
                # No current group - start new group for chunks without headings
                # Normalize "Untitled" labels so consecutive untitled chunks group together
                if chunk.section_label.startswith("Untitled"):
                    # Use page-based grouping for untitled chunks
                    current_header_path = f"Untitled (page {chunk.page_span[0]})"
                else:
                    current_header_path = chunk.section_label
                current_parent_chunks = [chunk]
    
    # Create parent from remaining group
    if current_parent_chunks:
        parent = _create_parent_from_chunks(
            current_parent_chunks,
            current_header_path or "Root",
            "heading_based"
        )
        parents.append(parent)
    
    return parents


def _create_soft_section_parents(
    chunks: List[Chunk],
    config
) -> List[ParentChunk]:
    """Create parents based on soft sections within page windows.
    
    Groups chunks by section_label, but merges "Untitled" chunks that are
    adjacent or in the same page window to avoid 1:1 parent-child relationships.
    Also groups consecutive chunks with the same normalized label.
    """
    parents = []
    page_window_size = config.parent_page_window_size
    
    # First pass: normalize section labels for grouping
    # Group consecutive "Untitled" chunks together
    current_group = []
    current_normalized_label = None
    
    for chunk in chunks:
        section_label = chunk.section_label
        
        # Normalize "Untitled" labels - group all "Untitled" chunks in same page together
        if section_label.startswith("Untitled"):
            normalized_label = f"Untitled (page {chunk.page_span[0]})"
        else:
            normalized_label = section_label
        
        # Check if we should start a new group
        if normalized_label != current_normalized_label and current_group:
            # Create parent from current group
            parent_label = current_normalized_label if current_normalized_label.startswith("Untitled") else current_group[0].section_label
            parent = _create_parent_from_chunks(
                current_group,
                parent_label,
                "soft_section"
            )
            parents.append(parent)
            current_group = []
        
        current_normalized_label = normalized_label
        current_group.append(chunk)
    
    # Create parent from remaining group
    if current_group:
        parent_label = current_normalized_label if current_normalized_label.startswith("Untitled") else current_group[0].section_label
        parent = _create_parent_from_chunks(
            current_group,
            parent_label,
            "soft_section"
        )
        parents.append(parent)
    
    return parents


def _group_chunks_by_page_window(
    chunks: List[Chunk],
    window_size: int
) -> List[List[Chunk]]:
    """Group chunks by page windows."""
    if not chunks:
        return []
    
    groups = []
    current_group = []
    current_page_start = None
    
    for chunk in chunks:
        page_start = chunk.page_span[0]
        
        if current_page_start is None:
            current_page_start = page_start
            current_group = [chunk]
        elif page_start - current_page_start < window_size:
            # Within window - add to current group
            current_group.append(chunk)
        else:
            # New window - start new group
            groups.append(current_group)
            current_page_start = page_start
            current_group = [chunk]
    
    # Add last group
    if current_group:
        groups.append(current_group)
    
    return groups


def _create_parent_from_chunks(
    chunks: List[Chunk],
    section_label: str,
    parent_type: str
) -> ParentChunk:
    """Create a parent chunk from a group of child chunks."""
    if not chunks:
        raise ValueError("Cannot create parent from empty chunk list")
    
    # Use first chunk as base
    first_chunk = chunks[0]
    
    # Combine content
    raw_parts = [chunk.raw_md_fragment for chunk in chunks]
    raw_md = "\n\n".join(raw_parts)
    
    text_parts = [chunk.text_for_embedding for chunk in chunks]
    text_embed = "\n\n".join(text_parts)
    
    # Combine metadata
    metadata = {
        "child_count": len(chunks),
        "parent_type": parent_type,
        "section_label": section_label,
    }
    
    # Get page span from all chunks
    page_nos = set()
    for chunk in chunks:
        page_nos.update(chunk.page_nos)
    
    page_nos_list = sorted(list(page_nos))
    page_span = (min(page_nos_list), max(page_nos_list))
    
    # Generate parent ID
    parent_id = generate_chunk_id(raw_md, {
        "doc_id": first_chunk.doc_id,
        "page_span": page_span,
        "section_label": section_label,
        "parent_type": parent_type
    })
    
    # Get child IDs
    child_ids = [chunk.chunk_id for chunk in chunks]
    
    # Merge line positions (min start, max end)
    line_starts = [chunk.line_start for chunk in chunks if chunk.line_start is not None]
    line_ends = [chunk.line_end for chunk in chunks if chunk.line_end is not None]
    line_start = min(line_starts) if line_starts else None
    line_end = max(line_ends) if line_ends else None
    
    from src.utils.tokens import count_tokens
    
    return ParentChunk(
        chunk_id=parent_id,
        doc_id=first_chunk.doc_id,
        page_span=page_span,
        page_nos=page_nos_list,
        header_path=first_chunk.header_path,
        section_label=section_label,
        element_type="parent",
        raw_md_fragment=raw_md,
        text_for_embedding=text_embed,
        metadata=metadata,
        parent_id=None,  # Parents don't have parents
        token_count=count_tokens(text_embed),
        node_id=first_chunk.node_id,
        child_ids=child_ids,
        parent_type=parent_type,
        line_start=line_start,
        line_end=line_end
    )


def assign_parent_ids(
    chunks: List[Chunk],
    parents: List[ParentChunk]
) -> List[Chunk]:
    """Assign parent_id to child chunks.
    
    Args:
        chunks: List of child chunks
        parents: List of parent chunks
    
    Returns:
        List[Chunk]: Chunks with assigned parent_ids
    """
    # Create mapping of child_id to parent_id
    child_to_parent = {}
    for parent in parents:
        for child_id in parent.child_ids:
            child_to_parent[child_id] = parent.chunk_id
    
    # Assign parent_ids
    for chunk in chunks:
        if chunk.chunk_id in child_to_parent:
            chunk.parent_id = child_to_parent[chunk.chunk_id]
    
    return chunks


def generate_stable_ids(chunks: List[Chunk]) -> List[Chunk]:
    """Generate stable IDs for chunks (already done in chunk creation, but ensure consistency).
    
    Args:
        chunks: List of chunks
    
    Returns:
        List[Chunk]: Chunks with stable IDs
    """
    # IDs are already generated during chunk creation using generate_chunk_id
    # This function ensures all chunks have IDs
    for chunk in chunks:
        if not chunk.chunk_id:
            chunk.chunk_id = generate_chunk_id(
                chunk.raw_md_fragment,
                {
                    "doc_id": chunk.doc_id,
                    "page_span": chunk.page_span,
                    "section_label": chunk.section_label
                }
            )
    
    return chunks

