# Chunking Pipeline

## Overview

The chunking pipeline intelligently splits documents into semantically meaningful chunks with hierarchical parent-child relationships. It preserves document structure while optimizing for token budgets.

## Main Function

### `process_document(file_path, config, doc_id)`

**Location**: `src/pipelines/chunking/chunking.py`

**Returns**: `(List[Chunk], List[ParentChunk], Dict[str, Any])`

**Pipeline Flow**:
1. Read Markdown file
2. Apply repair mode → `RepairResult`
3. Normalize page markers (`--- Page X ---`)
4. Split into `PageBlock`s with repair metadata
5. Merge continuations (if enabled and high confidence)
6. For each `PageBlock`:
   - Parse with LlamaIndex (`MarkdownNodeParser` → `HierarchicalNodeParser`)
   - Extract typed elements (`Heading`, `Paragraph`, `List`, `Table`, `ImageBlock`)
   - Generate `section_label`s
7. Structure-first chunking (create candidate chunks)
8. Token-budget refinement (split oversized, merge tiny)
9. Hybrid hierarchy building (parents + children)
10. Typed serialization (`raw_md_fragment` + `text_for_embedding`)
11. Generate stable IDs
12. Return `(children, parents, stats)`

## Markdown Repair

**Location**: `src/pipelines/chunking/repair.py`

### Function: `apply_repair_mode(content) -> RepairResult`

**Repair Operations**:
1. **Table Repair**: `repair_tables(lines)`
   - Detects table-like blocks (pipe density + alignment patterns)
   - If header separator missing: infers header from first row
   - If too broken: preserves as TableCandidate for chunking

2. **List Repair**: `repair_lists(lines)`
   - Detects list blocks by indentation/prefix patterns
   - Normalizes list markers
   - Treats each item as atomic (never splits items)

3. **Section Repair**: `repair_sections(lines)`
   - Detects and normalizes heading markers
   - Fixes broken heading hierarchies

**RepairResult Fields**:
- `repaired_content`: Repaired markdown string
- `repair_applied`: Dict mapping repair_type to list of RepairRecords
- `structure_confidence`: Confidence score (0.0-1.0)

## Page Block Processing

**Location**: `src/pipelines/chunking/page_parser.py`

### Function: `split_into_page_blocks(text, repair_result) -> List[PageBlock]`

**Process**:
- Detects page markers (`--- Page N ---`)
- Creates PageBlock for each page
- Assigns repair records to appropriate blocks
- Handles documents without page markers (single page)

### Function: `merge_continuations(blocks, config) -> List[PageBlock]`

**Continuation Detection**:
- **Paragraph continuation**: Ends mid-sentence or no blank line before marker
- **List continuation**: Ends in list context, next begins with same pattern
- **Table continuation**: Ends with table row, next begins with row-like pipes
- **Image continuation**: [IMAGE] not closed or sequential image blocks

**Aggressiveness Levels**: "low", "medium", "high"

**PageBlock Fields**:
- `page_no`: Page number
- `content`: Repaired content string
- `raw_lines`: List of raw lines
- `structure_confidence`: Confidence score (0.0-1.0)
- `repair_applied`: List of RepairRecord objects
- `page_span`: Tuple[int, int] - (start_page, end_page) for merged blocks

## LlamaIndex Integration

**Location**: `src/pipelines/chunking/llama_parser.py`

### Functions:
- `create_markdown_parser()`: Creates `MarkdownNodeParser`
- `create_hierarchical_parser()`: Creates `HierarchicalNodeParser`
- `parse_with_llamaindex(page_block, use_hierarchical)`: Parses page block

**Returns**: List of `BaseNode` objects with parent-child relationships

## Element Extraction

**Location**: `src/pipelines/chunking/element_extractor.py`

### Function: `extract_elements_from_nodes(nodes)`

**Process**:
1. Uses `markdown-it-py` for AST parsing
2. Traverses AST tokens
3. Extracts `Heading`, `Paragraph`, `ListElement`, `Table`, `ImageBlockElement`
4. Handles custom blocks ([IMAGE], [HEADER])
5. Builds header paths

**Element Types**:
- `Heading`: level, text, node_id, line_start, line_end
- `CustomHeader`: Custom header from [HEADER] tags
- `Paragraph`: Paragraph text with line positions
- `ListElement`: List with ordered, items, nesting
- `Table`: Table with raw_md, header_row, rows, signature
- `ImageBlockElement`: Image block with extracted text

## Structure-First Chunking

**Location**: `src/pipelines/chunking/chunkers.py`

### Class: `StructureFirstChunker`

**Purpose**: Creates candidate chunks from elements

**Process**:
- Groups elements by type and structure
- Creates chunks preserving document structure
- Handles prose, lists, tables, images separately

**Method**: `chunk(elements) -> List[Chunk]`

## Token-Budget Refinement

**Location**: `src/pipelines/chunking/chunkers.py`

### Class: `TokenBudgetRefiner`

**Purpose**: Refines chunks by token budget

**Process**:
- Splits oversized chunks (`> max_chunk_tokens_hard`)
- Merges tiny chunks (`< min_chunk_tokens`)
- Uses type-specific refiners

**Refiners**:
- `ProseRefiner`: Sentence-based splitting with windowing
- `ListRefiner`: Splits by list items
- `TableRefiner`: Splits by rows
- `ImageRefiner`: Handles image blocks

**Method**: `refine(chunks) -> List[Chunk]`

## Hybrid Hierarchy Building

**Location**: `src/pipelines/chunking/hierarchy.py`

### Function: `create_parents_hybrid(chunks, config) -> List[ParentChunk]`

**Strategy Selection**:
- Computes average structure confidence
- If `confidence >= threshold` → heading-based grouping
- Else → soft-section grouping

### Heading-Based Grouping
- Groups chunks by heading at target level (H2 by default)
- Chunks without headings grouped with preceding heading

### Soft-Section Grouping
- Groups by `section_label` within page windows
- Preserves element integrity
- Uses page window size for grouping

**ParentChunk Fields**:
- Inherits all `Chunk` fields
- `child_ids`: List of child chunk IDs
- `parent_type`: "heading_based" | "soft_section"

## Chunk Schema

**Location**: `src/schema/chunk.py`

### Chunk Fields:
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

## Configuration

**Location**: `src/config/chunking.py`

### ChunkConfig Fields:

**Prose Chunking**:
- `prose_target_tokens`: Target size (default: 512)
- `prose_overlap_tokens`: Overlap between chunks (default: 50)
- `sentence_window_size`: Sentences for windowing (default: 3)

**List Chunking**:
- `list_items_per_chunk`: Items per chunk (default: 10)
- `list_item_overlap`: Overlap items (default: 2)

**Table Chunking**:
- `table_rows_per_chunk`: Rows per chunk (default: 20)
- `table_row_overlap`: Overlap rows (default: 2)

**Hierarchy**:
- `parent_heading_level`: Heading level for parent grouping (default: 2 = H2)
- `parent_page_window_size`: Page window for soft sections (default: 3)
- `structure_confidence_threshold`: Threshold for heading-based vs soft-section (default: 0.6)

**Neighbors**:
- `neighbor_same_page`: Same-page neighbors for expansion (default: 1)
- `neighbor_cross_page`: Cross-page neighbors (default: 2)

**General**:
- `max_chunk_tokens_hard`: Hard limit (default: 2048)
- `min_chunk_tokens`: Minimum target size (default: 100)
- `enable_cross_page_merge`: Enable cross-page merging (default: True)
- `cross_page_merge_aggressiveness`: "low" | "medium" | "high" (default: "medium")

## Retrieval Safety

**Location**: `src/pipelines/chunking/retrieval_safety.py`

### Function: `expand_neighbors(chunk, all_chunks, config) -> List[Chunk]`

**Purpose**: Expands chunk with neighbors for retrieval safety

**Process**:
- Expands with same-page neighbors (siblings)
- Expands with cross-page neighbors (adjacent pages)
- Returns list including original chunk + neighbors

## Usage Example

```python
from src.config.chunking import ChunkConfig
from src.pipelines.chunking.chunking import process_document

config = ChunkConfig(
    prose_target_tokens=512,
    parent_heading_level=2,
    max_chunk_tokens_hard=2048
)

children, parents, stats = process_document("document.md", config, "doc_1")

print(f"Created {len(children)} children chunks")
print(f"Created {len(parents)} parent chunks")
print(f"Stats: {stats}")
```

## Next Steps

- **[Embedding & Indexing](./06-embedding-indexing.md)** - Index chunks to vector store
- **[Retrieval](./07-retrieval.md)** - Retrieve and use chunks
- **[Configuration](./08-configuration.md)** - Configure chunking settings

