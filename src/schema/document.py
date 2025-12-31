"""Unified document schema for ParseForge."""

import uuid
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field, field_validator


class BlockType(str, Enum):
    """Types of document blocks."""

    TEXT = "text"
    TITLE = "title"
    IMAGE = "image"
    TABLE = "table"
    LIST = "list"
    INDEX = "index"
    CODE = "code"
    INTERLINE_EQUATION = "interline_equation"
    INLINE_EQUATION = "inline_equation"
    CAPTION = "caption"
    FOOTNOTE = "footnote"
    HEADER = "header"
    FOOTER = "footer"
    PAGE_NUMBER = "page_number"


class BBox(BaseModel):
    """Normalized bounding box (0-1 coordinates)."""

    x0: float = Field(ge=0.0, le=1.0, description="Left coordinate (normalized)")
    y0: float = Field(ge=0.0, le=1.0, description="Top coordinate (normalized)")
    x1: float = Field(ge=0.0, le=1.0, description="Right coordinate (normalized)")
    y1: float = Field(ge=0.0, le=1.0, description="Bottom coordinate (normalized)")

    def to_tuple(self) -> Tuple[float, float, float, float]:
        """Convert to tuple format (x0, y0, x1, y1)."""
        return (self.x0, self.y0, self.x1, self.y1)

    def to_absolute(self, width: int, height: int) -> Tuple[int, int, int, int]:
        """Convert normalized coordinates to absolute pixel coordinates."""
        return (
            int(self.x0 * width),
            int(self.y0 * height),
            int(self.x1 * width),
            int(self.y1 * height),
        )

    @classmethod
    def from_absolute(
        cls, x0: int, y0: int, x1: int, y1: int, width: int, height: int
    ) -> "BBox":
        """Create from absolute pixel coordinates."""
        return cls(
            x0=x0 / width if width > 0 else 0.0,
            y0=y0 / height if height > 0 else 0.0,
            x1=x1 / width if width > 0 else 1.0,
            y1=y1 / height if height > 0 else 1.0,
        )

    @field_validator("x1", "y1")
    @classmethod
    def validate_coordinates(cls, v, info):
        """Ensure x1 > x0 and y1 > y0."""
        if info.field_name == "x1" and hasattr(info.data, "x0"):
            if v <= info.data.get("x0", 0):
                raise ValueError("x1 must be greater than x0")
        elif info.field_name == "y1" and hasattr(info.data, "y0"):
            if v <= info.data.get("y0", 0):
                raise ValueError("y1 must be greater than y0")
        return v


class Span(BaseModel):
    """Text or equation span within a block."""

    text: str
    bbox: Optional[BBox] = None
    span_type: str = "text"  # text, inline_equation, interline_equation
    confidence: Optional[float] = None
    language: Optional[str] = None


class Block(BaseModel):
    """Base block class for all document elements."""

    block_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    block_type: BlockType
    bbox: Optional[BBox] = None
    page_index: int
    page_range: Optional[Tuple[int, int]] = None
    confidence: Optional[float] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @field_validator("page_range")
    @classmethod
    def validate_page_range(cls, v):
        """Validate page range."""
        if v is None:
            return None
        start, end = v
        if start > end:
            raise ValueError("Start page must be <= end page")
        return v


class TextBlock(Block):
    """Text block containing paragraphs or text content."""

    text: str
    spans: List[Span] = Field(default_factory=list)
    language: Optional[str] = None


class TitleBlock(TextBlock):
    """Title or heading block."""

    level: int = Field(default=1, ge=1, le=6, description="Heading level (1-6)")


class ImageBlock(Block):
    """Image block."""

    image_path: Optional[str] = None
    image_data: Optional[bytes] = None
    caption: Optional[str] = None
    alt_text: Optional[str] = None


class TableBlock(Block):
    """Table block."""

    html: Optional[str] = None
    cells: List[List[str]] = Field(default_factory=list)
    headers: List[str] = Field(default_factory=list)
    num_rows: int = 0
    num_cols: int = 0


class ListBlock(Block):
    """List block (ordered or unordered)."""

    items: List[str] = Field(default_factory=list)
    ordered: bool = False


class CodeBlock(Block):
    """Code block."""

    code: str
    language: Optional[str] = None


class Page(BaseModel):
    """Page-level metadata and content."""

    page_index: int
    width: int
    height: int
    blocks: List[Block] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class Document(BaseModel):
    """Root document container."""

    file_name: Optional[str] = None
    file_path: Optional[str] = None
    pages: List[Page] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    total_pages: int = 0

    def get_all_blocks(self) -> List[Block]:
        """Get all blocks from all pages."""
        blocks = []
        for page in self.pages:
            blocks.extend(page.blocks)
        return blocks

