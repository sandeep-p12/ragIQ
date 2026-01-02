# Data Schemas

## Overview

RAG IQ uses Pydantic models for type-safe data structures throughout the system.

## Document Schema

**Location**: `src/schema/document.py`

### BlockType Enum

**Values**:
- `TEXT`: Plain text block
- `TITLE`: Title/heading block
- `IMAGE`: Image block
- `TABLE`: Table block
- `LIST`: List block
- `INDEX`: Table of contents/index
- `CODE`: Code block
- `INTERLINE_EQUATION`: Interline equation
- `INLINE_EQUATION`: Inline equation
- `CAPTION`: Caption block
- `FOOTNOTE`: Footnote block
- `HEADER`: Header block
- `FOOTER`: Footer block
- `PAGE_NUMBER`: Page number block

### BBox

**Purpose**: Normalized bounding box (0-1 coordinates)

**Fields**:
- `x0`, `y0`: Top-left corner (0-1)
- `x1`, `y1`: Bottom-right corner (0-1)

**Validation**: Ensures x1 > x0 and y1 > y0

### Span

**Purpose**: Text or equation span within a block

**Fields**:
- `text`: Text content
- `bbox`: Bounding box (optional)
- `is_equation`: Boolean

### Block (Base Class)

**Fields**:
- `block_id`: Unique block ID
- `block_type`: BlockType enum
- `bbox`: BBox
- `page_index`: Page index (0-based)
- `page_range`: Tuple[int, int] - (start_page, end_page)
- `confidence`: Confidence score (0-1)
- `metadata`: Additional metadata dict

### TextBlock(Block)

**Fields**:
- `text`: Text content
- `spans`: List of Span objects
- `language`: Language code (optional)

### TitleBlock(TextBlock)

**Fields**:
- `level`: Heading level (1-6)

### ImageBlock(Block)

**Fields**:
- `image_path`: Path to image file
- `image_data`: Base64 image data (optional)
- `caption`: Caption text (optional)
- `alt_text`: Alt text (optional)

### TableBlock(Block)

**Fields**:
- `html`: HTML table representation
- `cells`: 2D array of cell values
- `headers`: List of header strings
- `num_rows`: Number of rows
- `num_cols`: Number of columns

### ListBlock(Block)

**Fields**:
- `items`: List of list item strings
- `ordered`: Boolean (ordered vs unordered)

### CodeBlock(Block)

**Fields**:
- `code`: Code content
- `language`: Programming language

### Page

**Fields**:
- `page_index`: Page index (0-based)
- `width`: Page width in points
- `height`: Page height in points
- `blocks`: List of Block objects
- `metadata`: Additional metadata dict

### Document

**Fields**:
- `file_name`: Original filename
- `file_path`: File path
- `pages`: List of Page objects
- `metadata`: Additional metadata dict
- `total_pages`: Total number of pages

## Chunk Schema

**Location**: `src/schema/chunk.py`

### Element Types

**Heading**:
- `level`: Heading level (1-6)
- `text`: Heading text
- `node_id`: LlamaIndex node ID
- `line_start`, `line_end`: Line positions

**CustomHeader**:
- Custom header from [HEADER] tags

**Paragraph**:
- Paragraph text with line positions

**ListElement**:
- `ordered`: Boolean
- `items`: List of items
- `nesting`: Nesting level

**Table**:
- `raw_md`: Raw markdown
- `header_row`: Header row
- `rows`: Data rows
- `signature`: Table signature

**ImageBlockElement**:
- Image block with extracted text

**Element**: Union type of all elements

### Chunk Types

**RepairRecord**:
- `repair_type`: "table_repair", "list_repair", or "section_repair"
- `location`: (line_start, line_end) tuple
- `reason`: Description of repair
- `original`: Original content
- `repaired`: Repaired content

**RepairResult**:
- `repaired_content`: Repaired markdown string
- `repair_applied`: Dict mapping repair_type to list of RepairRecords
- `structure_confidence`: Confidence score (0.0-1.0)

**PageBlock**:
- `page_no`: Page number
- `content`: Repaired content string
- `raw_lines`: List of raw lines
- `start_line`: Start line number
- `end_line`: End line number
- `structure_confidence`: Confidence score (0.0-1.0)
- `repair_applied`: List of RepairRecord objects
- `page_span`: Tuple[int, int] - (start_page, end_page)
- `page_nos`: List[int] - All page numbers

**Chunk**:
- `chunk_id`: Stable chunk ID (SHA256 hash)
- `doc_id`: Document ID
- `page_span`: Tuple[int, int] - (start_page, end_page)
- `page_nos`: List[int] - All page numbers
- `header_path`: String path like "H1 > H2 > H3"
- `section_label`: Section label string
- `element_type`: Element type (heading, paragraph, list, table, image)
- `raw_md_fragment`: Original markdown (fidelity)
- `text_for_embedding`: Type-aware text for embedding
- `metadata`: Additional metadata dict
- `parent_id`: Parent chunk ID (optional)
- `token_count`: Token count
- `node_id`: LlamaIndex node ID
- `line_start`, `line_end`: Line positions

**ParentChunk(Chunk)**:
- `child_ids`: List of child chunk IDs
- `parent_type`: "heading_based" | "soft_section"

## Core Dataclasses

**Location**: `src/core/dataclasses.py`

### VectorRecord

**Fields**:
- `id`: Vector ID (chunk_id)
- `values`: List[float] - Embedding vector
- `metadata`: Dict - Chunk metadata

### Candidate

**Fields**:
- `chunk_id`: Chunk ID
- `score`: Similarity score
- `metadata`: Chunk metadata

### CandidateText

**Fields**:
- `chunk_id`: Chunk ID
- `text_snippet`: Text snippet
- `metadata`: Chunk metadata

### RerankResult

**Fields**:
- `chunk_id`: Chunk ID
- `relevance_score`: Score (0-100)
- `answerability`: Boolean
- `key_evidence`: Key evidence string

### ContextPack

**Fields**:
- `query`: Original query string
- `selected_chunks`: List of selected chunk dictionaries
- `citations`: List of citation strings
- `trace`: Dictionary with retrieval trace
- `context_pack`: Formatted context string

## Schema Validation

All schemas use Pydantic for:
- Type validation
- Automatic serialization/deserialization
- JSON schema generation
- Field validation

## Usage Examples

### Creating a Document

```python
from src.schema.document import Document, Page, TextBlock, BBox

page = Page(
    page_index=0,
    width=612,
    height=792,
    blocks=[
        TextBlock(
            block_id="block_1",
            block_type=BlockType.TEXT,
            bbox=BBox(x0=0.1, y0=0.1, x1=0.9, y1=0.2),
            page_index=0,
            text="Hello, world!"
        )
    ]
)

document = Document(
    file_name="example.pdf",
    file_path="/path/to/example.pdf",
    pages=[page]
)
```

### Working with Chunks

```python
from src.schema.chunk import Chunk

chunk = Chunk(
    chunk_id="chunk_1",
    doc_id="doc_1",
    page_span=(1, 2),
    page_nos=[1, 2],
    header_path="H1 > H2",
    section_label="Introduction",
    element_type="paragraph",
    raw_md_fragment="This is a paragraph.",
    text_for_embedding="This is a paragraph.",
    token_count=10,
    metadata={}
)
```

## Next Steps

- **[API Reference](./14-api-reference.md)** - Schema API details
- **[Parsing](./04-parsing.md)** - How documents are created
- **[Chunking](./05-chunking.md)** - How chunks are created

