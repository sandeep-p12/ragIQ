"""Retrieval safety: neighbor expansion rules for high-recall retrieval."""

from __future__ import annotations

from typing import List

from src.pipelines.chunking.chunkers import Chunk


def expand_neighbors(
    chunk: Chunk,
    all_chunks: List[Chunk],
    config
) -> List[Chunk]:
    """Expand chunk with neighbor chunks for retrieval.
    
    Includes:
    - Sibling children within same parent (±N, configurable)
    - Cross-page boundary neighbors (last N chunks from prev page + first N chunks from next page)
    - Prioritizes same element_type continuation (table/list/image)
    
    Args:
        chunk: Chunk to expand
        all_chunks: All available chunks
        config: ChunkConfig with neighbor settings
    
    Returns:
        List[Chunk]: Expanded chunk list with neighbors
    """
    expanded = [chunk]
    
    # 1. Sibling children within same parent
    if chunk.parent_id:
        siblings = _get_sibling_chunks(chunk, all_chunks, config.neighbor_same_page)
        expanded.extend(siblings)
    
    # 2. Cross-page boundary neighbors
    cross_page_neighbors = _get_cross_page_neighbors(
        chunk, all_chunks, config.neighbor_cross_page
    )
    expanded.extend(cross_page_neighbors)
    
    # Remove duplicates (keep original chunk first)
    seen_ids = {chunk.chunk_id}
    unique_expanded = [chunk]
    
    for neighbor in expanded[1:]:
        if neighbor.chunk_id not in seen_ids:
            seen_ids.add(neighbor.chunk_id)
            unique_expanded.append(neighbor)
    
    return unique_expanded


def _get_sibling_chunks(
    chunk: Chunk,
    all_chunks: List[Chunk],
    neighbor_count: int
) -> List[Chunk]:
    """Get sibling chunks within same parent.
    
    Args:
        chunk: Reference chunk
        all_chunks: All chunks
        neighbor_count: Number of neighbors to include (±N)
    
    Returns:
        List[Chunk]: Sibling chunks
    """
    if not chunk.parent_id:
        return []
    
    # Find all chunks with same parent_id
    siblings = [c for c in all_chunks if c.parent_id == chunk.parent_id]
    
    # Sort siblings (by page_span, then by some order)
    siblings.sort(key=lambda c: (c.page_span[0], c.page_span[1]))
    
    # Find index of current chunk
    try:
        current_idx = next(i for i, c in enumerate(siblings) if c.chunk_id == chunk.chunk_id)
    except StopIteration:
        return []
    
    # Get neighbors (±neighbor_count)
    start_idx = max(0, current_idx - neighbor_count)
    end_idx = min(len(siblings), current_idx + neighbor_count + 1)
    
    neighbors = siblings[start_idx:end_idx]
    # Remove the original chunk
    neighbors = [c for c in neighbors if c.chunk_id != chunk.chunk_id]
    
    return neighbors


def _get_cross_page_neighbors(
    chunk: Chunk,
    all_chunks: List[Chunk],
    neighbor_count: int
) -> List[Chunk]:
    """Get cross-page boundary neighbors.
    
    If chunk touches page boundary:
    - Include last N chunks from previous page
    - Include first N chunks from next page
    - Prioritize same element_type continuation
    
    Args:
        chunk: Reference chunk
        all_chunks: All chunks
        neighbor_count: Number of neighbors to include from each adjacent page
    
    Returns:
        List[Chunk]: Cross-page neighbor chunks
    """
    neighbors = []
    
    page_span = chunk.page_span
    page_start, page_end = page_span
    
    # Check if chunk touches page boundaries
    touches_start = page_start == min(c.page_span[0] for c in all_chunks if c.page_span[0] == page_start)
    touches_end = page_end == max(c.page_span[1] for c in all_chunks if c.page_span[1] == page_end)
    
    # Get chunks from previous page
    if page_start > 1 and not touches_start:
        prev_page_chunks = [
            c for c in all_chunks
            if c.page_span[1] == page_start - 1
        ]
        
        # Sort by position (approximate)
        prev_page_chunks.sort(key=lambda c: (c.page_span[0], c.page_span[1]))
        
        # Get last N chunks
        prev_neighbors = prev_page_chunks[-neighbor_count:] if len(prev_page_chunks) > neighbor_count else prev_page_chunks
        
        # Prioritize same element_type
        same_type = [c for c in prev_neighbors if c.element_type == chunk.element_type]
        other_type = [c for c in prev_neighbors if c.element_type != chunk.element_type]
        
        neighbors.extend(same_type)
        neighbors.extend(other_type[:neighbor_count - len(same_type)])
    
    # Get chunks from next page
    max_page = max(c.page_span[1] for c in all_chunks)
    if page_end < max_page and not touches_end:
        next_page_chunks = [
            c for c in all_chunks
            if c.page_span[0] == page_end + 1
        ]
        
        # Sort by position
        next_page_chunks.sort(key=lambda c: (c.page_span[0], c.page_span[1]))
        
        # Get first N chunks
        next_neighbors = next_page_chunks[:neighbor_count] if len(next_page_chunks) > neighbor_count else next_page_chunks
        
        # Prioritize same element_type
        same_type = [c for c in next_neighbors if c.element_type == chunk.element_type]
        other_type = [c for c in next_neighbors if c.element_type != chunk.element_type]
        
        neighbors.extend(same_type)
        neighbors.extend(other_type[:neighbor_count - len(same_type)])
    
    return neighbors

