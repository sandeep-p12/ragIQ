"""Chunking-related schemas: Chunk, ParentChunk, Element types, PageBlock, RepairRecord, RepairResult."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union


# Element types from element_extractor.py
@dataclass
class Heading:
    """Heading element."""
    level: int
    text: str
    node_id: str
    line_start: int
    line_end: int


@dataclass
class CustomHeader:
    """Custom header element from [HEADER]...[/HEADER]."""
    text: str
    node_id: str
    line_start: int
    line_end: int


@dataclass
class Paragraph:
    """Paragraph element."""
    text: str
    node_id: str
    line_start: int
    line_end: int


@dataclass
class ListElement:
    """List element (renamed from List to avoid conflict with built-in)."""
    ordered: bool
    items: List[str]
    nesting: str
    node_id: str
    line_start: int
    line_end: int


@dataclass
class Table:
    """Table element."""
    raw_md: str
    header_row: Optional[str]
    rows: List[str]
    signature: str
    node_id: str
    line_start: int
    line_end: int
    is_table_candidate: bool = False


@dataclass
class ImageBlockElement:
    """Image block element."""
    raw_text: str
    extracted_text: str
    node_id: str
    line_start: int
    line_end: int


# Union type for all elements
Element = Union[Heading, CustomHeader, Paragraph, ListElement, Table, ImageBlockElement]


# Repair dataclasses from repair.py
@dataclass
class RepairRecord:
    """Record of a repair operation."""
    repair_type: str  # table_repair, list_repair, section_repair
    location: Tuple[int, int]  # (line_start, line_end)
    reason: str
    original: str
    repaired: str


@dataclass
class RepairResult:
    """Result of repair operations."""
    repaired_content: str
    repair_applied: Dict[str, List[RepairRecord]] = field(default_factory=dict)
    structure_confidence: float = 1.0


# PageBlock from page_parser.py
@dataclass
class PageBlock:
    """Represents a page block with metadata."""
    page_no: int
    content: str  # Repaired content
    raw_lines: List[str]
    start_line: int
    end_line: int
    structure_confidence: float
    repair_applied: List[RepairRecord]
    page_span: Tuple[int, int] = None  # Will be set during merging
    page_nos: List[int] = None  # Will be set during merging
    
    def __post_init__(self):
        """Initialize page_span and page_nos after creation."""
        if self.page_span is None:
            self.page_span = (self.page_no, self.page_no)
        if self.page_nos is None:
            self.page_nos = [self.page_no]


# Chunk dataclass from chunkers.py
@dataclass
class Chunk:
    """Production chunk dataclass."""
    chunk_id: str
    doc_id: str
    page_span: Tuple[int, int]
    page_nos: List[int]
    header_path: Optional[str]
    section_label: str
    element_type: str
    raw_md_fragment: str  # Fidelity - original markdown
    text_for_embedding: str  # Type-aware serialization + minimal context
    metadata: Dict[str, Any] = field(default_factory=dict)
    parent_id: Optional[str] = None
    token_count: int = 0
    node_id: str = ""
    line_start: Optional[int] = None  # Start line in original document
    line_end: Optional[int] = None  # End line in original document


# ParentChunk from hierarchy.py
@dataclass
class ParentChunk(Chunk):
    """Parent chunk with child references."""
    child_ids: List[str] = field(default_factory=list)
    parent_type: str = "heading_based"  # "heading_based" or "soft_section"

