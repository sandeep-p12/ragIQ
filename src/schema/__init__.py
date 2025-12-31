"""Schema definitions for documents and chunks."""

from src.schema.chunk import (
    Chunk,
    CustomHeader,
    Element,
    Heading,
    ImageBlockElement,
    ListElement,
    PageBlock,
    ParentChunk,
    Paragraph,
    RepairRecord,
    RepairResult,
    Table,
)
from src.schema.document import (
    BBox,
    Block,
    BlockType,
    CodeBlock,
    Document,
    ImageBlock,
    ListBlock,
    Page,
    Span,
    TableBlock,
    TextBlock,
    TitleBlock,
)

__all__ = [
    # Document schema
    "BBox",
    "Block",
    "BlockType",
    "CodeBlock",
    "Document",
    "ImageBlock",
    "ListBlock",
    "Page",
    "Span",
    "TableBlock",
    "TextBlock",
    "TitleBlock",
    # Chunk schema
    "Chunk",
    "ParentChunk",
    "Element",
    "Heading",
    "CustomHeader",
    "Paragraph",
    "ListElement",
    "Table",
    "ImageBlockElement",
    "PageBlock",
    "RepairRecord",
    "RepairResult",
]

