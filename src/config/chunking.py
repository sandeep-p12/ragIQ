"""Configuration for chunking pipeline."""

from dataclasses import dataclass


@dataclass
class ChunkConfig:
    """Configuration for chunking pipeline."""
    prose_target_tokens: int = 512
    prose_overlap_tokens: int = 50
    sentence_window_size: int = 3
    list_items_per_chunk: int = 10
    list_item_overlap: int = 2
    table_rows_per_chunk: int = 20
    table_row_overlap: int = 2
    parent_heading_level: int = 2  # H2 by default
    parent_page_window_size: int = 3  # Fallback grouping window
    neighbor_same_page: int = 1  # ±1 sibling chunks
    neighbor_cross_page: int = 2  # N chunks from adjacent pages
    max_chunk_tokens_hard: int = 2048
    min_chunk_tokens: int = 100  # Minimum target size for chunks (tiny chunks below this will be merged aggressively)
    enable_cross_page_merge: bool = True
    cross_page_merge_aggressiveness: str = "medium"  # "low", "medium", "high"
    structure_confidence_threshold: float = 0.6  # For heading-based vs soft-section grouping

