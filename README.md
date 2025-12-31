# RAG IQ

**Production-grade RAG (Retrieval-Augmented Generation) system** with comprehensive document parsing, intelligent chunking, vector embedding, indexing, and retrieval capabilities.

## Table of Contents

1. [Overview](#overview)
2. [System Architecture](#system-architecture)
3. [Project Structure](#project-structure)
4. [Complete File Documentation](#complete-file-documentation)
5. [Data Flow and Control Flow](#data-flow-and-control-flow)
6. [Configuration and Environment Variables](#configuration-and-environment-variables)
7. [Module Dependencies](#module-dependencies)
8. [Installation](#installation)
9. [Usage](#usage)
10. [Component Interactions](#component-interactions)
11. [API Reference](#api-reference)

---

## Overview

RAG IQ is a complete end-to-end RAG pipeline that transforms documents into searchable, retrievable knowledge bases. The system processes documents through four main stages:

1. **Parsing**: Extracts structured content from various document formats (PDF, DOCX, PPTX, XLSX, CSV, HTML, TXT, MD)
2. **Chunking**: Intelligently splits documents into semantically meaningful chunks with hierarchical parent-child relationships
3. **Indexing**: Embeds chunks and stores them in vector databases (Pinecone, Azure AI Search)
4. **Retrieval**: Retrieves relevant chunks using vector similarity, reranks with LLM, and assembles context packs

### Key Features

- **Multi-format Document Parsing**: Supports 8+ document formats with intelligent strategy selection
- **Structure-First Chunking**: Preserves document structure while optimizing for token budgets
- **Hybrid Hierarchy**: Creates parent-child chunk relationships for better context assembly
- **LLM-Powered Reranking**: Uses GPT-4o to rerank retrieval candidates by relevance
- **Context Assembly**: Intelligently expands neighbors and assembles context packs within token budgets
- **Unified Streamlit UI**: Complete pipeline visualization and control interface

---

## System Architecture

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         Streamlit UI                             │
│  (streamlit_app.py) - Unified interface for all operations       │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    RAG Orchestrator                              │
│  (pipelines/orchestrator.py) - Coordinates all pipelines       │
└─────────────────────────────────────────────────────────────────┘
                              │
        ┌────────────────────┼────────────────────┐
        ▼                     ▼                     ▼
┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│   Parsing    │    │   Chunking    │    │  Retrieval   │
│   Pipeline   │───▶│   Pipeline    │───▶│   Pipeline   │
└──────────────┘    └──────────────┘    └──────────────┘
        │                     │                     │
        ▼                     ▼                     ▼
┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│  Providers   │    │   LlamaIndex │    │   Vector DB   │
│ (OCR, Layout)│    │   Parsers    │    │  (Pinecone)   │
└──────────────┘    └──────────────┘    └──────────────┘
```

### Component Layers

1. **UI Layer**: Streamlit application providing interactive interface
2. **Orchestration Layer**: Coordinates parsing, chunking, and retrieval
3. **Pipeline Layer**: Domain-specific pipelines (parsing, chunking, retrieval)
4. **Provider Layer**: External service integrations (OpenAI, Pinecone, OCR, Layout)
5. **Storage Layer**: Vector stores and chunk stores
6. **Schema Layer**: Data models and type definitions
7. **Config Layer**: Centralized configuration management

---

## Project Structure

```
ragIQ/
├── .env                          # Environment variables (create from template)
├── .gitignore                    # Git ignore rules
├── streamlit_app.py              # Unified Streamlit UI (1419 lines)
├── pyproject.toml                # Project dependencies and configuration
├── README.md                      # This comprehensive documentation
│
└── src/
    ├── __init__.py               # Package initialization
    │
    ├── core/                      # Core interfaces and dataclasses
    │   ├── __init__.py
    │   ├── base.py                # Base classes
    │   ├── dataclasses.py         # VectorRecord, Candidate, RerankResult, ContextPack
    │   └── interfaces.py          # Protocol-based interfaces (EmbeddingProvider, VectorStore, etc.)
    │
    ├── config/                     # Centralized configuration
    │   ├── __init__.py
    │   ├── chunking.py            # ChunkConfig dataclass
    │   ├── parsing.py             # ParseForgeConfig with Pydantic Settings
    │   ├── parsing_strategies.py  # StrategyEnum and strategy selection logic
    │   ├── prompts.py             # LLM prompts for parsing, images, tables
    │   └── retrieval.py          # RetrievalConfig, EmbeddingConfig, PineconeConfig, RerankConfig
    │
    ├── schema/                    # Document and chunk schemas
    │   ├── __init__.py
    │   ├── document.py            # Document, Page, Block types (Pydantic models)
    │   └── chunk.py               # Chunk, ParentChunk, Element types, PageBlock
    │
    ├── utils/                     # Utility functions
    │   ├── __init__.py
    │   ├── bbox.py                # Bounding box utilities
    │   ├── checkpoint.py           # Checkpoint management for parsing
    │   ├── env.py                  # Environment variable loading
    │   ├── exceptions.py           # Custom exceptions
    │   ├── ids.py                  # ID generation utilities
    │   ├── io.py                   # File I/O utilities (JSONL, citations)
    │   ├── memory.py               # Memory utilities
    │   ├── tokens.py               # Token counting (tiktoken)
    │   └── ui.py                   # UI formatting utilities
    │
    ├── providers/                 # External service providers
    │   ├── __init__.py
    │   ├── embedding/
    │   │   ├── __init__.py
    │   │   └── openai_embedding.py # OpenAIEmbeddingProvider with batching and caching
    │   ├── llm/
    │   │   ├── __init__.py
    │   │   └── openai_llm.py      # OpenAILLMProvider with vision support
    │   ├── ocr/
    │   │   ├── __init__.py
    │   │   └── doctr.py            # DoctrOCR provider for text detection
    │   └── layout/
    │       ├── __init__.py
    │       └── yolo.py             # YOLOLayoutDetector for layout detection
    │
    ├── storage/                    # Storage backends
    │   ├── __init__.py
    │   ├── chunk/
    │   │   ├── __init__.py
    │   │   ├── azure_blob.py       # Azure Blob Storage (future)
    │   │   └── local.py            # LocalChunkStore (file-based)
    │   └── vector/
    │       ├── __init__.py
    │       ├── azure_ai_search.py  # Azure AI Search (future)
    │       └── pinecone.py         # PineconeVectorStore
    │
    ├── formatters/                 # Output formatters
    │   ├── __init__.py
    │   ├── image.py                # ImageVisionLLMFormatter
    │   ├── markdown.py              # blocks_to_markdown converter
    │   └── table.py                 # TableLLMFormatter
    │
    ├── pipelines/                  # Pipeline orchestrators
    │   ├── __init__.py
    │   ├── orchestrator.py        # RAGOrchestrator (unified pipeline)
    │   │
    │   ├── parsing/                # Document parsing pipeline
    │   │   ├── __init__.py
    │   │   ├── parseforge.py       # ParseForge main orchestrator
    │   │   ├── parsers/
    │   │   │   ├── __init__.py
    │   │   │   ├── pdf.py          # PDFParser (816 lines) - Main PDF parser
    │   │   │   ├── docx.py         # DOCX parser
    │   │   │   ├── pptx.py         # PPTX parser
    │   │   │   ├── xlsx.py         # XLSX parser
    │   │   │   ├── csv.py          # CSV parser
    │   │   │   ├── html_txt_md.py  # HTML/TXT/MD parser
    │   │   │   ├── native_pdf_extractor.py  # Native PDF text extraction
    │   │   │   └── block_processor.py        # Block processing utilities
    │   │   └── processing/
    │   │       ├── __init__.py
    │   │       ├── doctr_ocr.py.bak # Backup OCR processor
    │   │       ├── magic.py         # MagicModel for table detection
    │   │       ├── para_split.py    # Paragraph splitting
    │   │       ├── reading_order.py # Reading order sorting
    │   │       ├── table_extractor.py # Table extraction
    │   │       └── table_merger.py  # Cross-page table merging
    │   │
    │   ├── chunking/               # Chunking pipeline
    │   │   ├── __init__.py
    │   │   ├── chunking.py          # Main process_document() function
    │   │   ├── chunkers.py          # StructureFirstChunker, TokenBudgetRefiner
    │   │   ├── element_extractor.py # Extract elements from LlamaIndex nodes
    │   │   ├── hierarchy.py         # create_parents_hybrid() - parent grouping
    │   │   ├── llama_parser.py      # LlamaIndex MarkdownNodeParser integration
    │   │   ├── page_parser.py       # Page block splitting and merging
    │   │   ├── repair.py            # Markdown repair operations
    │   │   └── retrieval_safety.py  # expand_neighbors() for retrieval expansion
    │   │
    │   └── retrieval/              # Retrieval pipeline
    │       ├── __init__.py
    │       ├── retrieval.py       # ingest_from_chunking_outputs(), retrieve()
    │       ├── reranker.py         # OpenAIReranker with strict JSON output
    │       └── context_assembler.py # DefaultContextAssembler with token budgeting
    │
    ├── ai_models/                  # AI model files (not in git)
    │   ├── crnn_vgg16_bn.pt        # Doctr recognition model
    │   ├── doclayout_yolo_ft.pt     # YOLO layout detection model
    │   └── fast_base.pt            # Doctr detection model
    │
    ├── data/                       # Runtime data directories
    │   ├── parsing/
    │   │   └── checkpoints/        # Parsing checkpoints for resume
    │   └── retrieval/
    │       ├── chunks/             # Local chunk storage (JSON files)
    │       └── parents/            # Local parent storage (JSON files)
    │
    └── tests/                      # Test files
        ├── __init__.py
        ├── test_chunking_pipeline.py
        ├── test_full_parsing_pipeline.py
        └── temp/                   # Temporary test outputs
```

---

## Complete File Documentation

### Root Files

#### `streamlit_app.py` (1419 lines)
**Purpose**: Unified Streamlit UI for the complete RAG pipeline.

**Key Functions**:
- `main()`: Entry point, creates tabbed interface
- `render_parse_tab()`: Document parsing UI with strategy selection
- `render_chunk_tab()`: Chunking UI with configuration options
- `render_index_tab()`: Indexing UI for vector store ingestion
- `render_retrieve_tab()`: Retrieval UI with query interface
- `render_pipeline_tab()`: End-to-end pipeline (placeholder)

**Session State Variables**:
- `document`: Parsed Document object
- `parsed_markdown`: Generated markdown string
- `parsed_file_name`: Original filename
- `chunks_children`: List of child chunks
- `chunks_parents`: List of parent chunks
- `processing_stats`: Chunking statistics
- `retrieval_result`: Last retrieval ContextPack
- `current_stage`: Current processing stage
- `progress`: Progress percentage (0.0-1.0)
- `stage_outputs`: Dictionary of stage outputs

**UI Features**:
- 5 main tabs: Parse, Chunk, Index, Retrieve, Pipeline
- Sidebar configuration panels for each stage
- Real-time progress bars and status updates
- Preview panels for markdown, chunks, and results
- Export/download functionality for all outputs
- Filtering and statistics visualization

#### `pyproject.toml`
**Purpose**: Project configuration, dependencies, and build settings.

**Key Sections**:
- `[project]`: Project metadata, Python version (>=3.10), dependencies
- `[project.optional-dependencies]`: Optional dependency groups (dev, onnx, doctr, torch, transformers, viz)
- `[build-system]`: Hatchling build backend
- `[tool.black]`: Black formatter configuration (line-length: 100)
- `[tool.ruff]`: Ruff linter configuration
- `[tool.pytest.ini_options]`: Pytest test configuration

**Core Dependencies**:
- `pydantic>=2.0.0`: Data validation
- `pydantic-settings>=2.0.0`: Settings management
- `tiktoken>=0.5.0`: Token counting
- `llama-index-core>=0.10.0`: Document parsing
- `langchain>=0.1.0`: Text splitting
- `openai>=1.0.0`: LLM and embeddings
- `pinecone>=3.0.0`: Vector database
- `streamlit>=1.28.0`: UI framework
- `pypdfium2>=4.30.0`: PDF processing
- `doclayout-yolo>=0.0.4`: Layout detection

### Core Module (`src/core/`)

#### `core/interfaces.py`
**Purpose**: Protocol-based interfaces for all providers and storage backends.

**Protocols Defined**:
- `EmbeddingProvider`: `embed_texts()`, `embed_query()`
- `VectorStore`: `upsert()`, `query()`
- `ChunkStore`: `put_chunks()`, `get_chunk()`, `get_chunks_bulk()`
- `Reranker`: `rerank()`
- `ContextAssembler`: `assemble()`
- `LLMProvider`: `generate()`
- `OCRProvider`: `detect_text()`
- `LayoutProvider`: `detect_layout()`

**Design Pattern**: Uses Python Protocols for duck typing, enabling easy swapping of implementations.

#### `core/dataclasses.py`
**Purpose**: Common dataclasses used across the system.

**Classes**:
- `VectorRecord`: `id`, `values` (List[float]), `metadata` (Dict)
- `Candidate`: `chunk_id`, `score`, `metadata`
- `CandidateText`: `chunk_id`, `text_snippet`, `metadata`
- `RerankResult`: `chunk_id`, `relevance_score` (0-100), `answerability`, `key_evidence`
- `ContextPack`: `query`, `selected_chunks`, `citations`, `trace`

#### `core/base.py`
**Purpose**: Base classes for common functionality.

**Status**: Currently empty - placeholder for future base classes if needed.

### Configuration Module (`src/config/`)

#### `config/parsing.py`
**Purpose**: ParseForge configuration using Pydantic Settings.

**Classes**:
- `LLMConfig`: LLM provider settings (`provider`, `model`, `api_key`)
- `StrategyConfig`: Strategy thresholds (`page_threshold`, `document_threshold`, finance variants)
- `ModelConfig`: Model file paths and architectures
- `ParseForgeConfig`: Main configuration class

**ParseForgeConfig Fields**:
- `device`: "cpu" | "cuda" | "mps" | "coreml"
- `batch_size`: Pages processed in parallel (default: 50)
- `page_threshold`: IoU threshold for page-level strategy (default: 0.6)
- `document_threshold`: Threshold for document-level strategy (default: 0.2)
- `finance_mode`: Enable stricter thresholds for financial documents
- `checkpoint_dir`: Path for parsing checkpoints
- `auto_resume`: Automatically resume from checkpoints
- `model_dir`: Directory containing AI models
- `yolo_layout_model`: YOLO model filename
- `doctr_det_arch`: Doctr detection architecture
- `doctr_reco_arch`: Doctr recognition architecture
- `llm_provider`: "openai" | "none"
- `llm_model`: Model name (default: "gpt-4o")
- `llm_api_key`: Optional API key override
- `llm_max_tokens`: Maximum tokens for LLM generation

**Environment Variable Prefix**: `PARSEFORGE_` (e.g., `PARSEFORGE_DEVICE`, `PARSEFORGE_LLM_API_KEY`)

#### `config/chunking.py`
**Purpose**: Chunking pipeline configuration.

**ChunkConfig Fields**:
- `prose_target_tokens`: Target size for prose chunks (default: 512)
- `prose_overlap_tokens`: Overlap between prose chunks (default: 50)
- `sentence_window_size`: Sentences to consider for windowing (default: 3)
- `list_items_per_chunk`: List items per chunk (default: 10)
- `list_item_overlap`: Overlap items between list chunks (default: 2)
- `table_rows_per_chunk`: Table rows per chunk (default: 20)
- `table_row_overlap`: Overlap rows between table chunks (default: 2)
- `parent_heading_level`: Heading level for parent grouping (default: 2 = H2)
- `parent_page_window_size`: Page window for soft sections (default: 3)
- `neighbor_same_page`: Same-page neighbors for expansion (default: 1)
- `neighbor_cross_page`: Cross-page neighbors for expansion (default: 2)
- `max_chunk_tokens_hard`: Hard limit for chunk tokens (default: 2048)
- `min_chunk_tokens`: Minimum target size (default: 100)
- `enable_cross_page_merge`: Enable cross-page merging (default: True)
- `cross_page_merge_aggressiveness`: "low" | "medium" | "high" (default: "medium")
- `structure_confidence_threshold`: Threshold for heading-based vs soft-section grouping (default: 0.6)

#### `config/retrieval.py`
**Purpose**: Retrieval pipeline configuration.

**Classes**:
- `EmbeddingConfig`: `model` (default: "text-embedding-3-small"), `batch_size` (100), `max_retries` (3)
- `PineconeConfig`: `api_key_env`, `index_name_env`, `namespace` ("children" | "parents"), `top_k_dense` (300)
- `RerankConfig`: `model` ("gpt-4o"), `max_candidates_to_rerank` (50), `return_top_n` (15), `max_text_chars_per_candidate` (1200), `strict_json_output` (True)
- `RetrievalConfig`: Main config with `neighbor_same_page`, `neighbor_cross_page`, `include_parents`, `final_max_tokens` (12000), `min_primary_hits_to_keep` (3)

**Environment Variables**:
- `PINECONE_API_KEY`: Pinecone API key
- `PINECONE_INDEX_NAME`: Pinecone index name (default: "hybrid-chunking")
- `PINECONE_NAMESPACE`: Namespace override

#### `config/parsing_strategies.py`
**Purpose**: Strategy selection logic for PDF parsing.

**StrategyEnum Values**:
- `FAST`: Native text extraction (pypdfium2)
- `HI_RES`: OCR-based extraction (Doctr)
- `AUTO`: Automatic selection per page
- `LLM_FULL`: Full document parsing using LLM vision

**Key Functions**:
- `get_page_strategy()`: Determines strategy for a single page based on IoU threshold
- `determine_global_strategy()`: Determines document-level strategy based on page strategies
- `PageStrategy`: Dataclass with `page_index` and `strategy`

**Logic**:
- Compares native text extraction vs OCR detection
- Uses IoU (Intersection over Union) to measure overlap
- If IoU < threshold → use HI_RES (OCR), else FAST (native)
- Document-level: If > threshold% pages need HI_RES → use HI_RES globally

#### `config/prompts.py`
**Purpose**: Centralized LLM prompts for parsing operations.

**Prompts**:
- `BASE_LLM_PROMPT`: Main prompt for PDF transcription with identifier tags ([TABLE], [IMAGE], [TOC], [HEADER])
- `IMAGE_DESCRIPTION_PROMPT`: Prompt for generating image descriptions
- `PAGE_PROCESSING_PROMPT`: Prompt for whole-page OCR and markdown generation
- `TABLE_EXTRACTION_PROMPT`: Prompt for extracting tables from images
- `TABLE_FORMATTING_PROMPT_TEMPLATE`: Template for formatting parsed tables
- `get_table_formatting_prompt()`: Function to generate table formatting prompts

### Schema Module (`src/schema/`)

#### `schema/document.py`
**Purpose**: Unified document schema using Pydantic models.

**Models**:
- `BlockType`: Enum (TEXT, TITLE, IMAGE, TABLE, LIST, INDEX, CODE, INTERLINE_EQUATION, INLINE_EQUATION, CAPTION, FOOTNOTE, HEADER, FOOTER, PAGE_NUMBER)
- `BBox`: Normalized bounding box (0-1 coordinates) with validation
- `Span`: Text or equation span within a block
- `Block`: Base block class with `block_id`, `block_type`, `bbox`, `page_index`, `page_range`, `confidence`, `metadata`
- `TextBlock(Block)`: Text content with `text`, `spans`, `language`
- `TitleBlock(TextBlock)`: Title with `level` (1-6)
- `ImageBlock(Block)`: Image with `image_path`, `image_data`, `caption`, `alt_text`
- `TableBlock(Block)`: Table with `html`, `cells`, `headers`, `num_rows`, `num_cols`
- `ListBlock(Block)`: List with `items`, `ordered`
- `CodeBlock(Block)`: Code with `code`, `language`
- `Page`: Page-level metadata with `page_index`, `width`, `height`, `blocks`, `metadata`
- `Document`: Root container with `file_name`, `file_path`, `pages`, `metadata`, `total_pages`

#### `schema/chunk.py`
**Purpose**: Chunking-related schemas and element types.

**Element Types**:
- `Heading`: `level`, `text`, `node_id`, `line_start`, `line_end`
- `CustomHeader`: Custom header from [HEADER] tags
- `Paragraph`: Paragraph text with line positions
- `ListElement`: List with `ordered`, `items`, `nesting`
- `Table`: Table with `raw_md`, `header_row`, `rows`, `signature`
- `ImageBlockElement`: Image block with extracted text
- `Element`: Union type of all elements

**Chunk Types**:
- `RepairRecord`: Repair operation record
- `RepairResult`: Result of repair operations
- `PageBlock`: Page block with content, raw_lines, repair metadata
- `Chunk`: Production chunk with `chunk_id`, `doc_id`, `page_span`, `page_nos`, `header_path`, `section_label`, `element_type`, `raw_md_fragment`, `text_for_embedding`, `metadata`, `parent_id`, `token_count`, `node_id`, `line_start`, `line_end`
- `ParentChunk(Chunk)`: Parent chunk with `child_ids`, `parent_type` ("heading_based" | "soft_section")

### Parsing Pipeline (`src/pipelines/parsing/`)

#### `parsing/parseforge.py`
**Purpose**: Main ParseForge orchestrator class.

**Class**: `ParseForge`

**Initialization**:
- Takes `ParseForgeConfig` and optional `progress_callback`
- Initializes `YOLOLayoutDetector`, `TableLLMFormatter`, `ImageVisionLLMFormatter`, `Checkpoint`

**Key Methods**:
- `parse(file_path, strategy, start_page, end_page)`: Main parsing method
  - Routes to format-specific parsers based on file extension
  - Returns `Document` object
- `to_markdown(document, generate_image_descriptions)`: Converts Document to markdown
  - Formats tables using LLM if enabled
  - Generates image descriptions if enabled
  - Returns markdown string
- `_format_tables_in_markdown()`: Formats tables in markdown using LLM
- `_update_progress()`: Calls progress callback

**Supported Formats**:
- `.pdf` → `PDFParser`
- `.docx` → `parse_docx()`
- `.pptx` → `parse_pptx()`
- `.xlsx`, `.xls` → `parse_xlsx()`
- `.csv` → `parse_csv()`
- `.html`, `.htm` → `parse_html()`
- `.txt`, `.md` → `parse_txt_md()`

#### `parsing/parsers/pdf.py` (816 lines)
**Purpose**: Comprehensive PDF parser with multiple strategies.

**Class**: `PDFParser`

**Initialization**:
- Initializes `DoctrOCR`, `YOLOLayoutDetector`, `ImageVisionLLMFormatter`
- Handles missing dependencies gracefully

**Key Methods**:
- `parse(file_path, strategy, start_page, end_page)`: Main parsing method
  - Handles `LLM_FULL` strategy separately
  - For other strategies:
    1. Rasterizes pages for strategy selection
    2. Determines page-level strategies (AUTO mode)
    3. Determines global strategy
    4. Processes pages in batches
    5. Applies post-processing (reading order, table merging, paragraph splitting)
    6. Returns `Document`
- `_parse_with_llm_full()`: Full document parsing using LLM vision
- `_process_page_batch()`: Processes a batch of pages
- `_process_page_fast()`: FAST strategy (native extraction)
- `_process_page_hi_res()`: HI_RES strategy (OCR + layout)

**Strategy Flow**:
1. **AUTO**: Runs OCR detection on all pages → determines per-page strategy → determines global strategy
2. **FAST**: Native text extraction → block processing → reading order
3. **HI_RES**: OCR detection → layout detection → block assembly → reading order
4. **LLM_FULL**: Vision LLM processes entire document

**Post-Processing**:
- Reading order sorting (`sort_blocks_by_reading_order()`)
- Cross-page table merging (`merge_cross_page_tables()`)
- Paragraph splitting (`split_paragraphs()`)

#### `parsing/parsers/native_pdf_extractor.py`
**Purpose**: Native PDF text extraction using pypdfium2.

**Function**: `extract_blocks_from_native_pdf(pdf_page)`
- Extracts text blocks with bounding boxes
- Handles text, images, and basic layout
- Returns list of blocks

#### `parsing/parsers/docx.py`
**Purpose**: DOCX parser for ParseForge.

**Function**: `parse_docx(file_path) -> Document`

**Implementation Details**:
- Uses `python-docx` library to read DOCX files
- Iterates through all paragraphs
- Determines block type based on paragraph style:
  - Styles containing "heading" or "title" → `TitleBlock` with level extracted from style name
  - Other paragraphs → `TextBlock`
- Extracts text content from paragraphs
- Creates single `Page` with all blocks (page_index=0)
- Standard page dimensions: 612x792 (letter size)
- Creates `Document` with single page
- Raises `ParserError` on failure

#### `parsing/parsers/pptx.py`
**Purpose**: PPTX parser for ParseForge.

**Function**: `parse_pptx(file_path) -> Document`

**Implementation Details**:
- Uses `python-pptx` library to read PPTX files
- Iterates through all slides
- Extracts text from shapes on each slide
- Detects title placeholders: `shape.placeholder_format.idx == 0` → `TitleBlock`
- Other text shapes → `TextBlock`
- Groups blocks by slide (page_index = slide index)
- Standard slide dimensions: 960x540
- Creates one `Page` per slide
- Creates `Document` with all pages
- Raises `ParserError` on failure

#### `parsing/parsers/xlsx.py`
**Purpose**: XLSX/XLS parser for ParseForge.

**Function**: `parse_xlsx(file_path) -> Document`

**Implementation Details**:
- Uses `openpyxl` library to read XLSX/XLS files
- Iterates through all sheets
- Converts each sheet to a `TableBlock`:
  - Extracts all rows using `iter_rows(values_only=True)`
  - Converts all cell values to strings
  - Stores in `cells` 2D array
  - Sets `num_rows` and `num_cols`
- Creates one `Page` per sheet (page_index = sheet index)
- Standard page dimensions: 612x792
- Creates `Document` with all pages
- Raises `ParserError` on failure

#### `parsing/parsers/csv.py`
**Purpose**: CSV parser for ParseForge.

**Function**: `parse_csv(file_path) -> Document`

**Implementation Details**:
- Uses `pandas` library to read CSV files
- Reads CSV into DataFrame
- Converts to `TableBlock`:
  - First row: column headers
  - Subsequent rows: data rows
  - Converts all values to strings
  - Stores in `cells` 2D array
  - Sets `num_rows` and `num_cols`
- Creates single `Page` (page_index=0)
- Standard page dimensions: 612x792
- Creates `Document` with single page
- Raises `ParserError` on failure

#### `parsing/parsers/html_txt_md.py`
**Purpose**: HTML, TXT, and MD file parsers.

**Functions**:

**`parse_html(file_path) -> Document`**:
- Uses BeautifulSoup to parse HTML
- Extracts text from common elements: `p`, `h1-h6`, `div`
- Determines block type:
  - Tags starting with `h` → `TitleBlock` with level from tag number
  - Other tags → `TextBlock`
- Strips whitespace and filters empty text
- Creates single `Page` (page_index=0)
- Standard page dimensions: 612x792
- Raises `ParserError` on failure

**`parse_txt_md(file_path) -> Document`**:
- Reads file line-by-line
- Detects markdown headings: lines starting with `#`
  - Level = number of `#` characters
  - Text = content after `#` and space
- Groups consecutive non-empty lines into paragraphs
- Creates `TitleBlock` for headings, `TextBlock` for paragraphs
- Preserves line breaks within paragraphs
- Creates single `Page` (page_index=0)
- Standard page dimensions: 612x792
- Raises `ParserError` on failure

#### `parsing/parsers/block_processor.py`
**Purpose**: Block processing utilities.

**Functions**:
- `fill_text_from_native()`: Fills text blocks from native extraction
- `fill_text_from_ocr()`: Fills text blocks from OCR detection
- `layout_detections_to_blocks()`: Converts layout detections to blocks

#### `parsing/processing/reading_order.py`
**Purpose**: Sort blocks by reading order using XY-Cut algorithm.

**Key Functions**:
- `sort_blocks_by_reading_order(blocks, page_width, page_height) -> List[Block]`: Main sorting function
  - Uses XY-Cut algorithm for complex layouts
  - Handles multi-column layouts
  - Preserves reading order (top-to-bottom, left-to-right)
  - Returns sorted list of blocks
- `recursive_xy_cut(boxes, axis) -> List[int]`: Recursive XY-Cut algorithm
  - Alternates between X and Y axes
  - Splits boxes by projection profiles
  - Returns sorted indices
- `projection_by_bboxes(boxes, axis) -> np.ndarray`: Creates projection profile
  - Projects bounding boxes onto axis
  - Returns 1D array of projections
- `split_projection_profile(profile, threshold) -> List[Tuple[int, int]]`: Splits profile
  - Finds valleys in projection profile
  - Identifies split points
  - Returns list of (start, end) ranges

#### `parsing/processing/table_extractor.py`
**Purpose**: Comprehensive table extraction from blocks and images.

**Key Functions**:
- `extract_table(block, page_width, page_height, use_vision_llm) -> TableBlock`: Main extraction function
  - Extracts table from TextBlock or ImageBlock
  - Uses OCR text or vision LLM for image-based tables
  - Builds table grid from text/OCR
  - Extracts cells and headers
  - Returns TableBlock with cells and HTML
- `extract_table_text(text, bbox, page_width, page_height) -> str`: Extracts table text
  - Splits text by bounding boxes
  - Identifies table-like structures
- `_split_table_text(text, bbox, page_width, page_height) -> List[Tuple[BBox, str]]`: Splits table text
  - Groups text by spatial proximity
  - Identifies rows and columns
- `build_table_grid(text_regions) -> List[List[str]]`: Builds table grid
  - Groups text regions into rows and columns
  - Handles merged cells
  - Returns 2D cell array
- `_extract_table_with_vision_llm(image_block, page_width, page_height) -> TableBlock`: Extracts table using vision LLM
  - Uses `TABLE_EXTRACTION_PROMPT`
  - Calls vision LLM to extract table from image
  - Parses markdown table response
  - Converts to TableBlock
- `_parse_markdown_table_to_cells(markdown_table) -> List[List[str]]`: Parses markdown table
  - Extracts cells from markdown table format
  - Handles separator rows
  - Returns 2D cell array
- `_cells_to_html(cells) -> str`: Converts cells to HTML table
  - Creates HTML table structure
  - Uses BeautifulSoup for formatting

#### `parsing/processing/table_merger.py`
**Purpose**: Merge tables that span multiple pages with header detection.

**Key Functions**:
- `merge_cross_page_tables(tables) -> List[TableBlock]`: Main merging function
  - Identifies continuation tables across pages
  - Merges cells and rows
  - Preserves headers
  - Returns merged table blocks
- `detect_table_headers(table) -> List[str]`: Detects table headers
  - Identifies header row (usually first row)
  - Returns list of header strings
- `can_merge_tables(table1, table2) -> bool`: Determines if two tables can be merged
  - Checks if headers match
  - Checks column count compatibility
  - Checks if table2 appears to be continuation
- `merge_tables(table1, table2) -> Optional[str]`: Merges two tables
  - Combines cells and rows
  - Preserves headers from first table
  - Returns merged HTML or None if merge failed

#### `parsing/processing/para_split.py`
**Purpose**: Paragraph splitting with list and index detection.

**Key Functions**:
- `split_paragraphs(blocks) -> List[Block]`: Main splitting function
  - Splits long paragraphs into smaller blocks
  - Preserves paragraph structure
  - Handles lists and indexes separately
  - Returns list of blocks
- `is_list_block(block) -> bool`: Detects if block is a list
  - Checks for list markers (-, *, +, numbered)
  - Checks indentation patterns
- `is_index_block(block) -> bool`: Detects if block is a table of contents/index
  - Checks for index-like patterns (numbered items with page references)
  - Checks for TOC markers
- `merge_text_blocks(block1, block2) -> TextBlock`: Merges two text blocks
  - Combines text content
  - Preserves metadata from first block
  - Updates bounding boxes

#### `parsing/processing/magic.py`
**Purpose**: Magic model for table detection with confidence filtering.

**Class**: `MagicModel`

**Key Methods**:
- `__init__(detections, page_width, page_height)`: Initializes with layout detections
- `get_detections() -> List[LayoutDetectionOutput]`: Returns filtered detections
  - Applies axis fixing (removes invalid detections)
  - Removes low confidence detections
  - Removes high IoU duplicates
  - Returns cleaned list of detections
- `_fix_axis()`: Fixes invalid bounding boxes
  - Ensures x1 > x0 and y1 > y0
  - Clips to page boundaries
- `_remove_low_confidence()`: Removes detections below confidence threshold
- `_remove_high_iou_duplicates()`: Removes duplicate detections
  - Uses IoU threshold to identify duplicates
  - Keeps highest confidence detection

### Chunking Pipeline (`src/pipelines/chunking/`)

#### `chunking/chunking.py`
**Purpose**: Main chunking pipeline orchestration.

**Function**: `process_document(file_path, config, doc_id) -> (List[Chunk], List[ParentChunk], Dict)`

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

**Returns**:
- `children`: List of child `Chunk` objects
- `parents`: List of `ParentChunk` objects
- `stats`: Dictionary with processing statistics

#### `chunking/chunkers.py`
**Purpose**: Structure-first chunking and token-budget refinement.

**Classes**:
- `StructureFirstChunker`: Creates candidate chunks from elements
  - Groups elements by type and structure
  - Creates chunks preserving document structure
  - Handles prose, lists, tables, images separately
- `TokenBudgetRefiner`: Refines chunks by token budget
  - Splits oversized chunks (`> max_chunk_tokens_hard`)
  - Merges tiny chunks (`< min_chunk_tokens`)
  - Uses type-specific refiners
- `ProseRefiner`: Refines prose chunks
  - Sentence-based splitting with windowing
  - LangChain fallback for complex cases
- `ListRefiner`: Refines list chunks
  - Splits by list items
  - Preserves list structure
- `TableRefiner`: Refines table chunks
  - Splits by rows
  - Preserves table structure
- `ImageRefiner`: Refines image chunks
  - Handles image blocks

**Key Methods**:
- `StructureFirstChunker.chunk()`: Creates chunks from elements
- `TokenBudgetRefiner.refine()`: Refines chunks by token budget
- `_merge_tiny_chunks()`: Merges adjacent tiny chunks

#### `chunking/element_extractor.py`
**Purpose**: Extract typed elements from LlamaIndex nodes using AST parsing.

**Key Functions**:
- `extract_elements_from_nodes(nodes)`: Main extraction function
  - Uses `markdown-it-py` for AST parsing
  - Extracts `Heading`, `Paragraph`, `ListElement`, `Table`, `ImageBlockElement`
  - Handles custom blocks ([IMAGE], [HEADER])
- `build_header_path_stack(nodes)`: Builds header path hierarchy
  - Creates paths like "H1 > H2 > H3"
  - Used for section labeling

**Element Extraction Logic**:
1. Parse markdown with `markdown-it-py`
2. Traverse AST tokens
3. Identify headings, paragraphs, lists, tables
4. Extract text and metadata
5. Build header paths

#### `chunking/llama_parser.py`
**Purpose**: LlamaIndex integration for markdown parsing.

**Functions**:
- `create_markdown_parser()`: Creates `MarkdownNodeParser`
- `create_hierarchical_parser()`: Creates `HierarchicalNodeParser`
- `parse_with_llamaindex(page_block, use_hierarchical)`: Parses page block
  - Returns list of `BaseNode` objects
  - Hierarchical parser creates parent-child relationships

#### `chunking/page_parser.py`
**Purpose**: Page-aware processing with safe cross-page continuation merging.

**Key Functions**:
- `detect_page_markers(text) -> List[Tuple[int, int, str]]`: Detects page markers
  - Finds normalized page markers (`--- Page N ---`)
  - Returns list of (line_number, page_number, marker_text) tuples
- `split_into_page_blocks(text, repair_result) -> List[PageBlock]`: Splits text into PageBlocks
  - Detects page markers
  - Creates PageBlock for each page
  - Assigns repair records to appropriate blocks
  - Handles documents without page markers (single page)
  - Returns list of PageBlock objects with metadata
- `detect_continuation(page_block, next_block, aggressiveness) -> bool`: Detects continuation
  - **Paragraph continuation**: Ends mid-sentence or no blank line before marker
  - **List continuation**: Ends in list context, next begins with same pattern
  - **Table continuation**: Ends with table row, next begins with row-like pipes
  - **Image continuation**: [IMAGE] not closed or sequential image blocks
  - Aggressiveness levels: "low", "medium", "high"
  - Returns True if continuation detected
- `merge_continuations(blocks, config) -> List[PageBlock]`: Merges page blocks
  - Only merges when high-confidence heuristics trigger
  - Preserves `page_span` and `page_nos` metadata
  - Combines repair records and structure confidence
  - Respects `enable_cross_page_merge` config setting
  - Returns merged blocks

**PageBlock Dataclass**:
- `page_no`: Page number
- `content`: Repaired content string
- `raw_lines`: List of raw lines
- `start_line`: Start line number in original document
- `end_line`: End line number in original document
- `structure_confidence`: Confidence score (0.0-1.0)
- `repair_applied`: List of RepairRecord objects
- `page_span`: Tuple[int, int] - (start_page, end_page) for merged blocks
- `page_nos`: List[int] - All page numbers covered by this block

#### `chunking/repair.py`
**Purpose**: Markdown repair mode with table, list, and section repair with metadata tracking.

**Key Functions**:
- `apply_repair_mode(content) -> RepairResult`: Main repair function
  - Applies all repair operations
  - Tracks repair records
  - Computes structure confidence
  - Returns `RepairResult` with repaired content and metadata
- `normalize_page_markers(text) -> str`: Normalizes page markers
  - Supports `--- Page N ---` and `-- page N --` formats
  - Normalizes to `--- Page N ---` format
  - Case and whitespace tolerant
- `repair_tables(lines) -> (List[str], List[RepairRecord])`: Repairs malformed tables
  - Detects table-like blocks (pipe density + alignment patterns)
  - If header separator missing: infers header from first row, synthesizes separator
  - If too broken: preserves as TableCandidate for chunking
  - Returns repaired lines and repair records
- `repair_lists(lines) -> (List[str], List[RepairRecord])`: Repairs malformed lists
  - Detects list blocks by indentation/prefix patterns
  - Normalizes list markers (converts to consistent format)
  - Treats each item as atomic (never splits items)
  - Returns repaired lines and repair records
- `repair_sections(lines) -> (List[str], List[RepairRecord])`: Repairs section markers
  - Detects and normalizes heading markers
  - Fixes broken heading hierarchies
  - Returns repaired lines and repair records

**RepairRecord Fields**:
- `repair_type`: "table_repair", "list_repair", or "section_repair"
- `location`: (line_start, line_end) tuple
- `reason`: Description of repair
- `original`: Original content
- `repaired`: Repaired content

**RepairResult Fields**:
- `repaired_content`: Repaired markdown string
- `repair_applied`: Dict mapping repair_type to list of RepairRecords
- `structure_confidence`: Confidence score (0.0-1.0) based on repairs applied

#### `chunking/hierarchy.py`
**Purpose**: Hybrid parent grouping (heading-based or soft sections).

**Function**: `create_parents_hybrid(chunks, config) -> List[ParentChunk]`

**Strategy Selection**:
- Computes average structure confidence
- If `confidence >= threshold` → heading-based grouping
- Else → soft-section grouping

**Heading-Based Grouping**:
- Groups chunks by heading at target level (H2 by default)
- Chunks without headings grouped with preceding heading

**Soft-Section Grouping**:
- Groups by `section_label` within page windows
- Preserves element integrity
- Uses page window size for grouping

**Functions**:
- `_create_heading_based_parents()`: Creates parents from headings
- `_create_soft_section_parents()`: Creates parents from soft sections
- `_create_parent_from_chunks()`: Creates `ParentChunk` from child chunks
- `assign_parent_ids()`: Assigns parent IDs to children
- `generate_stable_ids()`: Generates stable chunk IDs

#### `chunking/retrieval_safety.py`
**Purpose**: Neighbor expansion for retrieval safety.

**Function**: `expand_neighbors(chunk, all_chunks, config) -> List[Chunk]`
- Expands chunk with same-page neighbors (siblings)
- Expands with cross-page neighbors (adjacent pages)
- Returns list including original chunk + neighbors

**Functions**:
- `_get_sibling_chunks()`: Gets sibling chunks within same parent
- `_get_cross_page_neighbors()`: Gets neighbors from adjacent pages

### Retrieval Pipeline (`src/pipelines/retrieval/`)

#### `retrieval/retrieval.py`
**Purpose**: High-level retrieval pipeline functions.

**Functions**:
- `ingest_from_chunking_outputs(children_jsonl_path, parents_jsonl_path, doc_id, cfg)`: Ingests chunks to vector store
  1. Loads JSONL files
  2. Stores chunks locally (`LocalChunkStore`)
  3. Embeds children texts (`OpenAIEmbeddingProvider`)
  4. Creates `VectorRecord`s with metadata
  5. Upserts to Pinecone (children and optionally parents)
  6. Returns ingestion stats
- `retrieve(query, filters, cfg, all_chunks) -> ContextPack`: Retrieves and assembles context
  1. Embeds query
  2. Queries Pinecone (`top_k_dense` candidates)
  3. Fetches candidate text snippets from chunk store
  4. Reranks candidates with LLM
  5. Assembles context pack with neighbor expansion
  6. Returns `ContextPack`

#### `retrieval/reranker.py`
**Purpose**: OpenAI LLM reranker with strict JSON output.

**Class**: `OpenAIReranker`

**Key Methods**:
- `rerank(query, candidates) -> List[RerankResult]`
  1. Builds prompt with query and candidates
  2. Calls LLM with JSON response format
  3. Parses and validates response
  4. Returns sorted results (by relevance_score descending)
- `_build_prompt()`: Builds reranking prompt
  - Includes candidate metadata (chunk_id, section_label, page_span, element_type)
  - Includes text snippets (trimmed to `max_text_chars_per_candidate`)
- `_call_llm_with_retry()`: Calls LLM with exponential backoff
- `_parse_response()`: Parses and validates JSON response
  - Uses Pydantic models (`RerankItem`, `RerankResponse`)
  - Retries with repair prompt if invalid

**Scoring Rubric**:
- 100: Directly answers query with explicit evidence
- 70: Highly relevant but partial answer
- 40: Tangential context
- 0: Unrelated

**Caching**: Results cached by query + candidate IDs

#### `retrieval/context_assembler.py`
**Purpose**: Context assembler with hierarchy-aware expansion and token budgeting.

**Class**: `DefaultContextAssembler`

**Method**: `assemble(query, reranked, cfg, doc_id, candidate_metadata, all_chunks) -> ContextPack`

**Assembly Flow**:
1. **Start with reranked top results** (`return_top_n`)
2. **Fetch primary chunks** from chunk store
3. **Include parent chunks** (if `include_parents=True`)
4. **Expand neighbors** using `expand_neighbors()`
   - Same-page neighbors (siblings)
   - Cross-page neighbors
5. **Ordering and deduplication**:
   - Primary reranked chunks (highest score first)
   - Their parent chunks (immediately after each child)
   - Same-parent neighbors (siblings)
   - Cross-page neighbors (last)
6. **Apply token budget**:
   - Always keep top `min_primary_hits_to_keep` primary chunks
   - Drop in priority order: cross_page > sibling > parent > primary
   - Stop when `final_max_tokens` reached
7. **Convert to final format**:
   - Create chunk dictionaries
   - Generate citations
   - Build trace

**Returns**: `ContextPack` with `selected_chunks`, `citations`, `trace`

### Providers (`src/providers/`)

#### `providers/embedding/openai_embedding.py`
**Purpose**: OpenAI embedding provider with batching and retry logic.

**Class**: `OpenAIEmbeddingProvider`

**Key Methods**:
- `embed_texts(texts) -> List[List[float]]`: Embeds batch of texts
  - Checks cache first
  - Batches uncached texts
  - Embeds with retry logic
  - Caches results
- `embed_query(query) -> List[float]`: Embeds single query
- `_embed_batch_with_retry()`: Embeds batch with exponential backoff

**Features**:
- Caching by model + text
- Batch processing (`batch_size` from config)
- Retry logic with exponential backoff
- Rate limit handling

#### `providers/llm/openai_llm.py`
**Purpose**: OpenAI LLM provider with vision support.

**Class**: `OpenAILLMProvider`

**Key Methods**:
- `generate(prompt, model, temperature, max_tokens) -> str`: Text generation
- `generate_vision(prompt, images, model, temperature, max_tokens) -> str`: Vision generation
  - Converts images to base64
  - Supports PIL Images, base64 strings, bytes

**Features**:
- Handles missing API key gracefully
- Rate limit and API error handling
- Vision support for image processing

#### `providers/ocr/doctr.py`
**Purpose**: Doctr OCR provider for text detection.

**Class**: `DoctrOCR`

**Key Methods**:
- `detect_text(images) -> List[TextDetection]`: Detects text in images
  - Uses Doctr detection and recognition models
  - Returns text detections with bounding boxes

**TextDetection Dataclass**:
- `text`: Detected text
- `bbox`: Bounding box
- `confidence`: Detection confidence
- `page_index`: Page index
- `dimensions`: Image dimensions

#### `providers/layout/yolo.py`
**Purpose**: YOLO-based layout detection using doclayout_yolo YOLOv10.

**Class**: `YOLOLayoutDetector`

**Initialization**:
- Loads YOLO model from `doclayout_yolo_ft.pt`
- Supports devices: cpu, cuda, mps, coreml
- Configurable confidence threshold (default: 0.2)
- Image size: 1024 for prediction

**Label Mapping** (doclayout_yolo_ft.pt model):
- 0: "title" → `BlockType.TITLE`
- 1: "plain text" → `BlockType.TEXT`
- 2: "abandon" → skipped
- 3: "figure" → `BlockType.IMAGE`
- 4: "figure_caption" → `BlockType.CAPTION`
- 5: "table" → `BlockType.TABLE`
- 6: "table_caption" → `BlockType.CAPTION`
- 7: "table_footnote" → `BlockType.FOOTNOTE`
- 8: "isolate_formula" → `BlockType.TEXT`
- 9: "formula_caption" → `BlockType.TEXT`

**Key Methods**:
- `detect_layout(image, page_index) -> List[LayoutDetectionOutput]`: Detects layout elements
  - Preprocesses image (resize, normalize)
  - Runs YOLO inference
  - Filters by confidence threshold
  - Converts detections to `LayoutDetectionOutput` with bounding boxes
  - Maps labels to `BlockType` enum

**LayoutDetectionOutput Dataclass**:
- `bbox`: Normalized bounding box (BBox)
- `category_id`: YOLO category ID
- `score`: Confidence score
- `block_type`: Mapped BlockType

### Storage (`src/storage/`)

#### `storage/vector/pinecone.py`
**Purpose**: Pinecone vector store implementation.

**Class**: `PineconeVectorStore`

**Key Methods**:
- `__init__(config)`: Initializes Pinecone client
  - Creates index if doesn't exist (serverless or pod-based)
  - Determines dimension from embedding model
- `upsert(vectors, namespace)`: Upserts vectors to Pinecone
  - Batches vectors (100 at a time)
  - Sanitizes metadata (JSON-serializable)
- `query(vector, top_k, filters, namespace) -> List[Candidate]`: Queries Pinecone
  - Builds Pinecone filter from metadata filters
  - Returns `Candidate` objects with scores

**Features**:
- Automatic index creation
- Metadata sanitization
- Batch upserting
- Filter support

#### `storage/chunk/local.py`
**Purpose**: Local file-based chunk store.

**Class**: `LocalChunkStore`

**Key Methods**:
- `put_chunks(children, parents)`: Stores chunks to disk
  - Creates directory structure: `chunks/{doc_id}/{chunk_id}.json`
  - Atomic writes (temp file + rename)
- `get_chunk(doc_id, chunk_id, is_parent) -> Dict`: Gets single chunk
- `get_chunks_bulk(keys) -> List[Dict]`: Gets multiple chunks

**Storage Structure**:
```
src/data/retrieval/
├── chunks/
│   └── {doc_id}/
│       └── {chunk_id}.json
└── parents/
    └── {doc_id}/
        └── {chunk_id}.json
```

#### `storage/chunk/azure_blob.py`
**Purpose**: Azure Blob Storage chunk store placeholder (TODO: implement for future migration).

**Class**: `AzureBlobStore(ChunkStore)`

**Status**: Not yet implemented - placeholder for future Azure Blob Storage integration.

**Interface**: Matches `ChunkStore` protocol for easy migration:
- `put_chunks(children, parents) -> None`
- `get_chunk(doc_id, chunk_id, is_parent) -> Optional[Dict]`
- `get_chunks_bulk(keys) -> List[Optional[Dict]]`

**Note**: Currently raises `NotImplementedError`. Use `LocalChunkStore` for now.

#### `storage/vector/azure_ai_search.py`
**Purpose**: Azure AI Search vector store placeholder (TODO: implement for future migration).

**Class**: `AzureAISearchStore(VectorStore)`

**Status**: Not yet implemented - placeholder for future Azure AI Search integration.

**Interface**: Matches `VectorStore` protocol for easy migration:
- `upsert(vectors, namespace) -> None`
- `query(vector, top_k, filters, namespace) -> List[Candidate]`

**Note**: Currently raises `NotImplementedError`. Use `PineconeVectorStore` for now.

### Formatters (`src/formatters/`)

#### `formatters/markdown.py`
**Purpose**: Convert Document blocks to markdown with comprehensive formatting.

**Key Functions**:
- `blocks_to_markdown(blocks, image_formatter, image_descriptions) -> str`: Main conversion function
- `format_title_block(block) -> str`: Formats title blocks with heading levels (#)
- `format_text_block(block) -> str`: Formats text blocks with whitespace normalization
- `format_list_block(block) -> str`: Formats list blocks (ordered/unordered)
- `format_table_block(block) -> str`: Formats table blocks with cell sanitization
  - Extracts LLM-formatted markdown from HTML comments
  - Falls back to HTML conversion using BeautifulSoup
  - Sanitizes table cells (removes pipes, normalizes whitespace)
  - Ensures consistent column counts
- `format_image_block(block, image_formatter, image_descriptions) -> str`: Formats image blocks
  - Adds image descriptions if available
  - Includes captions and alt text
- `format_code_block(block) -> str`: Formats code blocks with language tags
- `escape_markdown(text) -> str`: Escapes special markdown characters
- `sanitize_table_cell(cell) -> str`: Sanitizes table cells for markdown compatibility
- `_clean_markdown_table(table_markdown) -> str`: Cleans and validates markdown tables

#### `formatters/table.py`
**Purpose**: Table formatting using text-only LLM.

**Class**: `TableLLMFormatter`

**Key Methods**:
- `format_table(table, previous_table) -> TableBlock`: Formats table block using LLM
  - Extracts table text from TableBlock
  - Calls LLM with formatting prompt
  - Extracts markdown table from response
  - Stores markdown in HTML with markers for markdown formatter
  - Returns updated TableBlock
- `_extract_table_text(table) -> str`: Extracts text representation from TableBlock
- `_extract_markdown_table(content) -> str`: Extracts markdown table from LLM response
  - Removes code block markers
  - Finds lines starting with `|`
- `_markdown_to_html(markdown) -> str`: Converts markdown table to HTML (for backward compatibility)

#### `formatters/image.py`
**Purpose**: Vision LLM formatter for image descriptions and whole-page processing.

**Class**: `ImageVisionLLMFormatter`

**Key Methods**:
- `describe_image(image_block) -> str`: Generates description for a single image
  - Filters out icons/decorative images (size-based heuristics)
  - Uses `IMAGE_DESCRIPTION_PROMPT`
  - Calls OpenAI vision API with image
  - Returns detailed description or empty string if skipped
- `describe_images_batch(image_blocks) -> List[str]`: Batch processing for multiple images
- `format_description_tag(description) -> str`: Formats description with `[IMAGE_DESCRIPTION: ...]` tag
- `process_page_with_images(page_image, image_blocks, page_index) -> dict`: Processes whole page
  - Uses `PAGE_PROCESSING_PROMPT` for OCR + image descriptions
  - Maintains layout and positions
  - Extracts image descriptions from markdown
  - Returns OCR markdown, image descriptions dict, and blocks
- `_is_important_image(image_block, image) -> bool`: Determines if image is important
  - Checks size (filters small icons)
  - Checks aspect ratio
  - Checks for captions/alt text with important keywords
- `_image_to_base64(image) -> str`: Converts PIL Image to base64
- `_load_image(image_block) -> Optional[Image]`: Loads image from ImageBlock

### Utilities (`src/utils/`)

#### `utils/env.py`
**Purpose**: Environment variable loading.

**Functions**:
- `load_env() -> Dict[str, str]`: Loads all environment variables from `.env`
- `get_openai_api_key() -> Optional[str]`: Gets OpenAI API key

#### `utils/tokens.py`
**Purpose**: Token counting and structure confidence utilities.

**Key Functions**:
- `count_tokens(text, model) -> int`: Counts tokens in text using tiktoken
  - Uses model-specific encoding (default: "gpt-3.5-turbo")
  - Falls back to "cl100k_base" if model not found
  - Returns token count
- `compute_structure_confidence(repair_records, page_content) -> float`: Computes heuristic structure confidence
  - Base confidence: 1.0
  - Penalizes based on number of repairs (0.1 per repair, max 0.3)
  - Penalizes by repair type: table_repair (-0.15), list_repair (-0.1), section_repair (-0.05)
  - Boosts confidence if content has headings (+0.1)
  - Returns value between 0.0 and 1.0
  - Used for determining heading-based vs soft-section grouping

#### `utils/ids.py`
**Purpose**: ID generation utilities for deterministic chunk IDs.

**Function**: `generate_chunk_id(content, metadata) -> str`
- Generates deterministic chunk ID using SHA256 hash
- Uses chunk content (raw_md_fragment) and metadata
- Creates stable string representation with sorted keys
- Returns 16-character hexadecimal hash
- Ensures same content + metadata = same ID

#### `utils/io.py`
**Purpose**: I/O utilities for JSONL and citation formatting.

**Key Functions**:
- `load_jsonl(file_path) -> List[Dict[str, Any]]`: Loads JSONL file
  - Reads line-by-line
  - Parses each line as JSON
  - Returns list of dictionaries
  - Handles empty lines gracefully
- `save_jsonl(data, file_path) -> None`: Saves list of dictionaries to JSONL file
  - Writes each dictionary as one line
  - Uses UTF-8 encoding
  - Ensures ASCII-safe output
- `format_citation(chunk) -> str`: Formats human-readable citation string
  - Includes: Doc ID, Page span, Section label/header path
  - Adds element-specific ranges (row_range, item_range) if available
  - Format: `"Doc: {doc_id} | Page: {page} | Section: {section}"`

#### `utils/checkpoint.py`
**Purpose**: Checkpoint/resume mechanism for ParseForge parsing.

**Class**: `Checkpoint`

**Initialization**:
- Takes `checkpoint_dir` Path
- Creates directory if doesn't exist

**Key Methods**:
- `save(document_path, last_batch, last_page, strategy, batches, checkpoint_id) -> str`: Saves checkpoint
  - Saves to JSON file: `{checkpoint_id}.json`
  - Stores version, document path, batch/page progress, strategy, batch statuses
  - Returns checkpoint file path
- `load(checkpoint_id) -> Dict[str, Any]`: Loads checkpoint
  - Loads JSON file
  - Returns checkpoint data dictionary
  - Raises `CheckpointError` if not found
- `list_checkpoints() -> List[str]`: Lists all checkpoint IDs
  - Returns list of checkpoint file stems
- `delete(checkpoint_id) -> None`: Deletes a checkpoint
  - Removes checkpoint file

**Checkpoint Data Structure**:
```json
{
  "version": "1.0",
  "document_path": "...",
  "last_batch": 0,
  "last_page": 5,
  "strategy": "AUTO",
  "batches": [...]
}
```

#### `utils/exceptions.py`
**Purpose**: Exception hierarchy for ParseForge.

**Exception Classes**:
- `ParseForgeException`: Base exception for all ParseForge errors
- `StrategyException(ParseForgeException)`: Exception raised during strategy selection
- `OCRError(ParseForgeException)`: Exception raised during OCR processing
- `LayoutError(ParseForgeException)`: Exception raised during layout detection
- `TableError(ParseForgeException)`: Exception raised during table processing
- `ParserError(ParseForgeException)`: Exception raised during document parsing
- `CheckpointError(ParseForgeException)`: Exception raised during checkpoint operations
- `ConfigurationError(ParseForgeException)`: Exception raised due to configuration issues

#### `utils/ui.py`
**Purpose**: UI utility functions for Streamlit.

**Key Functions**:
- `display_error(error, details) -> None`: Displays error in Streamlit UI
  - Shows error message and optional details
  - Displays exception traceback
- `format_progress_message(stage, progress) -> str`: Formats progress message
  - Converts progress (0-1) to percentage
  - Returns formatted string
- `format_stage_output(stage, output_data) -> Dict[str, Any]`: Formats output data for display
  - Handles multiple stage types:
    - `file_loading`: File name, size, page count, type
    - `strategy_selection`: Total pages, FAST/HI_RES counts, page strategies
    - `ocr_processing`: Pages processed, text blocks, confidence
    - `layout_detection`: Element counts, detected labels, model classes
    - `post_processing`: Removed items, duplicates, footnotes, overlaps
    - `reading_order`: Pages processed, blocks sorted, ordering method
    - `table_extraction`: Tables extracted, rows, columns, cells
    - `table_merging`: Tables merged, continuations, operations
    - `paragraph_splitting`: Paragraphs, lists, block types
    - `llm_formatting`: Tables/images processed, API calls, provider
    - `markdown_generation`: Markdown length, blocks, tables, images
  - Returns formatted dictionary for UI display

### Orchestrator (`src/pipelines/orchestrator.py`)

#### `pipelines/orchestrator.py`
**Purpose**: Unified orchestrator for complete RAG pipeline.

**Class**: `RAGOrchestrator`

**Initialization**:
- Takes optional `parsing_config`, `chunking_config`, `retrieval_config`, `progress_callback`
- Initializes `ParseForge` parser

**Key Methods**:
- `parse(file_path, strategy, start_page, end_page) -> Document`: Parse document
- `parse_to_markdown(file_path, strategy, generate_image_descriptions, start_page, end_page) -> str`: Parse to markdown
- `chunk(markdown_path, doc_id) -> (List[Chunk], List[ParentChunk], Dict)`: Chunk markdown
- `index(children, parents, doc_id, output_dir) -> Dict`: Index chunks
- `query(query, filters, all_chunks) -> ContextPack`: Query indexed documents
- `pipeline(file_path, doc_id, strategy, generate_image_descriptions, index_chunks) -> Dict`: Complete pipeline

**Pipeline Method Flow**:
1. Parse document
2. Convert to markdown
3. Save markdown to temp file
4. Chunk markdown
5. Index chunks (if requested)
6. Return results dictionary

---

## Data Flow and Control Flow

### End-to-End Data Flow

```
Document File (PDF/DOCX/etc.)
    │
    ▼
[ParseForge.parse()]
    │
    ├─► Format Detection
    │
    ├─► Strategy Selection (AUTO/FAST/HI_RES/LLM_FULL)
    │
    ├─► Page Processing
    │   ├─► Native Extraction (FAST)
    │   ├─► OCR Detection (HI_RES)
    │   ├─► Layout Detection (HI_RES)
    │   └─► Vision LLM (LLM_FULL)
    │
    ├─► Post-Processing
    │   ├─► Reading Order Sorting
    │   ├─► Table Merging
    │   └─► Paragraph Splitting
    │
    ▼
Document Object (Pydantic Model)
    │
    ▼
[ParseForge.to_markdown()]
    │
    ├─► Block-to-Markdown Conversion
    │
    ├─► Table Formatting (LLM)
    │
    ├─► Image Description Generation (LLM Vision)
    │
    ▼
Markdown String
    │
    ▼
[process_document()]
    │
    ├─► Markdown Repair
    │
    ├─► Page Block Splitting
    │
    ├─► LlamaIndex Parsing
    │
    ├─► Element Extraction
    │
    ├─► Structure-First Chunking
    │
    ├─► Token-Budget Refinement
    │
    ├─► Hierarchy Building
    │
    ▼
(List[Chunk], List[ParentChunk], Stats)
    │
    ▼
[ingest_from_chunking_outputs()]
    │
    ├─► Local Chunk Storage
    │
    ├─► Text Embedding (OpenAI)
    │
    ├─► Vector Record Creation
    │
    └─► Pinecone Upsert
    │
    ▼
Indexed Vector Database
    │
    ▼
[retrieve()]
    │
    ├─► Query Embedding
    │
    ├─► Pinecone Query (top_k_dense)
    │
    ├─► Candidate Text Fetching
    │
    ├─► LLM Reranking
    │
    ├─► Neighbor Expansion
    │
    ├─► Context Assembly
    │
    └─► Token Budgeting
    │
    ▼
ContextPack (selected_chunks, citations, trace)
```

### Parsing Control Flow

```
PDFParser.parse()
    │
    ├─► [LLM_FULL Strategy?]
    │   └─► _parse_with_llm_full()
    │       └─► Vision LLM processes entire document
    │
    ├─► [Other Strategies]
    │   │
    │   ├─► Load PDF Document
    │   │
    │   ├─► Rasterize Pages
    │   │
    │   ├─► [AUTO Strategy?]
    │   │   ├─► Run OCR Detection on All Pages
    │   │   ├─► Determine Per-Page Strategy
    │   │   └─► Determine Global Strategy
    │   │
    │   ├─► Process Pages in Batches
    │   │   │
    │   │   ├─► [FAST Strategy]
    │   │   │   ├─► Native Text Extraction
    │   │   │   ├─► Block Processing
    │   │   │   └─► Reading Order
    │   │   │
    │   │   └─► [HI_RES Strategy]
    │   │       ├─► OCR Detection
    │   │       ├─► Layout Detection
    │   │       ├─► Block Assembly
    │   │       └─► Reading Order
    │   │
    │   ├─► Post-Processing
    │   │   ├─► Reading Order Sorting
    │   │   ├─► Cross-Page Table Merging
    │   │   └─► Paragraph Splitting
    │   │
    │   └─► Return Document
```

### Chunking Control Flow

```
process_document()
    │
    ├─► Read Markdown File
    │
    ├─► Apply Repair Mode
    │   └─► RepairResult (repaired_content, repair_applied)
    │
    ├─► Normalize Page Markers
    │   └─► Replace "--- Page X ---" markers
    │
    ├─► Split into PageBlocks
    │   └─► List[PageBlock] (page_no, content, repair_metadata)
    │
    ├─► Merge Continuations (if enabled)
    │   └─► Merge high-confidence continuation blocks
    │
    ├─► For Each PageBlock:
    │   │
    │   ├─► Parse with LlamaIndex
    │   │   ├─► MarkdownNodeParser
    │   │   └─► HierarchicalNodeParser
    │   │
    │   ├─► Extract Elements
    │   │   ├─► AST Parsing (markdown-it-py)
    │   │   ├─► Extract Headings, Paragraphs, Lists, Tables
    │   │   └─► Build Header Paths
    │   │
    │   └─► Generate Section Labels
    │
    ├─► Structure-First Chunking
    │   └─► StructureFirstChunker.chunk()
    │       ├─► Group Elements by Type
    │       ├─► Create Candidate Chunks
    │       └─► Preserve Structure
    │
    ├─► Token-Budget Refinement
    │   └─► TokenBudgetRefiner.refine()
    │       ├─► Split Oversized Chunks
    │       └─► Merge Tiny Chunks
    │
    ├─► Hybrid Hierarchy Building
    │   └─► create_parents_hybrid()
    │       ├─► Compute Structure Confidence
    │       ├─► [High Confidence?]
    │       │   └─► Heading-Based Grouping
    │       └─► [Low Confidence?]
    │           └─► Soft-Section Grouping
    │
    ├─► Typed Serialization
    │   ├─► raw_md_fragment (fidelity)
    │   └─► text_for_embedding (type-aware)
    │
    ├─► Generate Stable IDs
    │
    └─► Return (children, parents, stats)
```

### Retrieval Control Flow

```
retrieve()
    │
    ├─► Embed Query
    │   └─► OpenAIEmbeddingProvider.embed_query()
    │
    ├─► Query Pinecone
    │   └─► PineconeVectorStore.query()
    │       └─► Returns top_k_dense Candidates
    │
    ├─► Fetch Candidate Texts
    │   └─► LocalChunkStore.get_chunk()
    │       └─► Trim to max_text_chars_per_candidate
    │
    ├─► Rerank Candidates
    │   └─► OpenAIReranker.rerank()
    │       ├─► Build Prompt
    │       ├─► Call LLM (JSON response)
    │       ├─► Parse Response
    │       └─► Sort by relevance_score
    │
    ├─► Assemble Context
    │   └─► DefaultContextAssembler.assemble()
    │       ├─► Fetch Primary Chunks
    │       ├─► Include Parent Chunks
    │       ├─► Expand Neighbors
    │       │   ├─► Same-Page Neighbors
    │       │   └─► Cross-Page Neighbors
    │       ├─► Order and Deduplicate
    │       ├─► Apply Token Budget
    │       └─► Generate Citations
    │
    └─► Return ContextPack
```

---

## Configuration and Environment Variables

### Environment Variables

All configuration is loaded from the `.env` file in the project root. Environment variables use prefixes to organize settings.

#### ParseForge Configuration (Prefix: `PARSEFORGE_`)

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `PARSEFORGE_DEVICE` | str | `"cpu"` | Device: "cpu", "cuda", "mps", "coreml" |
| `PARSEFORGE_BATCH_SIZE` | int | `50` | Pages processed in parallel |
| `PARSEFORGE_PAGE_THRESHOLD` | float | `0.6` | IoU threshold for page-level strategy |
| `PARSEFORGE_DOCUMENT_THRESHOLD` | float | `0.2` | Threshold for document-level strategy |
| `PARSEFORGE_FINANCE_MODE` | bool | `False` | Enable finance mode (stricter thresholds) |
| `PARSEFORGE_FINANCE_PAGE_THRESHOLD` | float | `0.7` | Finance mode page threshold |
| `PARSEFORGE_FINANCE_DOCUMENT_THRESHOLD` | float | `0.15` | Finance mode document threshold |
| `PARSEFORGE_CHECKPOINT_DIR` | str | `"src/data/parsing/checkpoints"` | Checkpoint directory |
| `PARSEFORGE_AUTO_RESUME` | bool | `True` | Auto-resume from checkpoints |
| `PARSEFORGE_MODEL_DIR` | str | `"src/ai_models"` | AI models directory |
| `PARSEFORGE_YOLO_LAYOUT_MODEL` | str | `"doclayout_yolo_ft.pt"` | YOLO model filename |
| `PARSEFORGE_DOCTR_DET_ARCH` | str | `"fast_base"` | Doctr detection architecture |
| `PARSEFORGE_DOCTR_RECO_ARCH` | str | `"crnn_vgg16_bn"` | Doctr recognition architecture |
| `PARSEFORGE_LLM_PROVIDER` | str | `"openai"` | LLM provider: "openai" or "none" |
| `PARSEFORGE_LLM_MODEL` | str | `"gpt-4o"` | LLM model name |
| `PARSEFORGE_LLM_API_KEY` | str | - | LLM API key (optional override) |
| `PARSEFORGE_LLM_MAX_TOKENS` | int | `1000` | Maximum tokens for LLM generation |

#### OpenAI Configuration

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `OPENAI_API_KEY` | str | - | **Required** OpenAI API key for embeddings and LLM |

#### Pinecone Configuration

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `PINECONE_API_KEY` | str | - | **Required** Pinecone API key |
| `PINECONE_INDEX_NAME` | str | `"hybrid-chunking"` | Pinecone index name |
| `PINECONE_NAMESPACE` | str | `"children"` | Pinecone namespace ("children" or "parents") |

### Configuration Objects

#### ParseForgeConfig
Created from environment variables with `PARSEFORGE_` prefix. Can also be instantiated directly:

```python
from src.config.parsing import ParseForgeConfig

config = ParseForgeConfig(
    device="cuda",
    batch_size=100,
    llm_provider="openai",
    llm_model="gpt-4o"
)
```

#### ChunkConfig
Created directly (no environment variables):

```python
from src.config.chunking import ChunkConfig

config = ChunkConfig(
    prose_target_tokens=512,
    prose_overlap_tokens=50,
    parent_heading_level=2,
    max_chunk_tokens_hard=2048
)
```

#### RetrievalConfig
Created from environment with `from_env()`:

```python
from src.config.retrieval import RetrievalConfig

config = RetrievalConfig.from_env()
# Or with overrides:
config = RetrievalConfig.from_env(
    neighbor_same_page=2,
    final_max_tokens=15000
)
```

### Example `.env` File

```bash
# OpenAI
OPENAI_API_KEY=sk-...

# Pinecone
PINECONE_API_KEY=...
PINECONE_INDEX_NAME=hybrid-chunking
PINECONE_NAMESPACE=children

# ParseForge
PARSEFORGE_DEVICE=cpu
PARSEFORGE_BATCH_SIZE=50
PARSEFORGE_PAGE_THRESHOLD=0.6
PARSEFORGE_DOCUMENT_THRESHOLD=0.2
PARSEFORGE_LLM_PROVIDER=openai
PARSEFORGE_LLM_MODEL=gpt-4o
PARSEFORGE_LLM_MAX_TOKENS=1000
PARSEFORGE_AUTO_RESUME=True
```

---

## Module Dependencies

### External Dependencies

#### Core
- `pydantic>=2.0.0`: Data validation and settings
- `pydantic-settings>=2.0.0`: Settings management
- `python-dotenv>=1.0.0`: Environment variable loading
- `numpy>=1.24.0`: Numerical operations
- `tiktoken>=0.5.0`: Token counting

#### Document Parsing
- `pillow>=10.0.0`: Image processing
- `pypdfium2>=4.30.0`: PDF processing
- `beautifulsoup4>=4.12.0`: HTML parsing
- `python-docx>=1.1.0`: DOCX parsing
- `python-pptx>=0.6.23`: PPTX parsing
- `openpyxl>=3.1.0`: XLSX parsing
- `pandas>=2.0.0`: Data manipulation

#### Chunking
- `llama-index-core>=0.10.0`: Document parsing
- `llama-index-readers-file>=0.1.0`: File readers
- `langchain>=0.1.0`: Text splitting
- `langchain-community>=0.0.20`: Community integrations
- `markdown-it-py>=3.0.0`: Markdown parsing

#### Embedding and LLM
- `openai>=1.0.0`: OpenAI API client

#### Vector Store
- `pinecone>=3.0.0`: Pinecone client

#### UI
- `streamlit>=1.28.0`: Streamlit framework

#### Layout Detection
- `doclayout-yolo>=0.0.4`: YOLO layout detection
- `opencv-python-headless>=4.8.0`: Image processing
- `huggingface-hub>=0.20.0`: Model loading
- `onnxruntime>=1.16.0`: ONNX runtime
- `onnxtr[cpu]>=0.6.0`: ONNX transformers

### Internal Dependencies

```
streamlit_app.py
    ├─► src.config.*
    ├─► src.pipelines.*
    ├─► src.schema.*
    └─► src.utils.*

pipelines/orchestrator.py
    ├─► config.*
    ├─► pipelines.parsing.parseforge
    ├─► pipelines.chunking.chunking
    └─► pipelines.retrieval.retrieval

pipelines/parsing/parseforge.py
    ├─► config.parsing
    ├─► providers.layout.yolo
    ├─► formatters.*
    ├─► parsers.*
    └─► schema.document

pipelines/chunking/chunking.py
    ├─► config.chunking
    ├─► chunkers.*
    ├─► element_extractor.*
    ├─► hierarchy.*
    ├─► llama_parser.*
    ├─► page_parser.*
    └─► repair.*

pipelines/retrieval/retrieval.py
    ├─► config.retrieval
    ├─► providers.embedding.openai_embedding
    ├─► storage.vector.pinecone
    ├─► storage.chunk.local
    ├─► retrieval.reranker
    └─► retrieval.context_assembler
```

---

## Installation

### Prerequisites

- Python >= 3.10
- `uv` package manager (recommended) or `pip`

### Step 1: Install `uv` (if not installed)

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Step 2: Install Dependencies

```bash
uv sync
```

Or with pip:

```bash
pip install -e .
```

### Step 3: Configure Environment

Create a `.env` file in the project root:

```bash
cp .env.example .env  # If example exists
# Or create manually
```

Add your API keys:

```bash
OPENAI_API_KEY=sk-...
PINECONE_API_KEY=...
```

### Step 4: Download AI Models (Optional)

Place model files in `src/ai_models/`:
- `doclayout_yolo_ft.pt`: YOLO layout detection model
- `crnn_vgg16_bn.pt`: Doctr recognition model
- `fast_base.pt`: Doctr detection model

If models are missing, the system will disable those features gracefully.

### Step 5: Run Streamlit UI

```bash
streamlit run streamlit_app.py
```

The UI will open at `http://localhost:8501`.

---

## Usage

### Streamlit UI

The Streamlit UI provides five main tabs:

1. **Parse**: Upload and parse documents
   - Select parsing strategy (AUTO, FAST, HI_RES, LLM_FULL)
   - Configure device, batch size, thresholds
   - Enable finance mode for stricter thresholds
   - Configure LLM settings
   - Preview parsed markdown

2. **Chunk**: Chunk parsed or uploaded markdown
   - Configure prose, list, and table chunking
   - Set hierarchy parameters
   - Configure neighbor expansion
   - Preview chunks with filters

3. **Index**: Index chunks to vector store
   - Configure embedding settings
   - Set Pinecone index and namespace
   - View indexing statistics

4. **Retrieve**: Query and retrieve chunks
   - Enter query
   - Configure reranking settings
   - Set retrieval parameters
   - View retrieval results with trace

5. **Pipeline**: End-to-end pipeline (placeholder)

### Programmatic Usage

#### Parse Document

```python
from src.config.parsing import ParseForgeConfig
from src.config.parsing_strategies import StrategyEnum
from src.pipelines.parsing.parseforge import ParseForge

config = ParseForgeConfig(device="cpu", llm_provider="openai")
parser = ParseForge(config)

document = parser.parse("document.pdf", strategy=StrategyEnum.AUTO)
markdown = parser.to_markdown(document, generate_image_descriptions=True)
```

#### Chunk Document

```python
from src.config.chunking import ChunkConfig
from src.pipelines.chunking.chunking import process_document

config = ChunkConfig(
    prose_target_tokens=512,
    parent_heading_level=2
)

children, parents, stats = process_document("document.md", config, "doc_1")
```

#### Index Chunks

```python
from src.config.retrieval import RetrievalConfig
from src.pipelines.retrieval.retrieval import ingest_from_chunking_outputs

config = RetrievalConfig.from_env()

stats = ingest_from_chunking_outputs(
    "children.jsonl",
    "parents.jsonl",
    "doc_1",
    config
)
```

#### Retrieve

```python
from src.config.retrieval import RetrievalConfig
from src.pipelines.retrieval.retrieval import retrieve

config = RetrievalConfig.from_env()

result = retrieve(
    "What is the main topic?",
    filters={"doc_id": "doc_1"},
    cfg=config
)

print(f"Retrieved {len(result.selected_chunks)} chunks")
print(result.context_pack)
```

#### Complete Pipeline

```python
from src.pipelines.orchestrator import RAGOrchestrator
from src.config.parsing_strategies import StrategyEnum

orchestrator = RAGOrchestrator()

results = orchestrator.pipeline(
    "document.pdf",
    "doc_1",
    strategy=StrategyEnum.AUTO,
    generate_image_descriptions=True,
    index_chunks=True
)

print(f"Parsed {len(results['document'].pages)} pages")
print(f"Created {len(results['children'])} children chunks")
print(f"Created {len(results['parents'])} parent chunks")
```

---

## Component Interactions

### Parsing → Chunking

**Data Flow**:
- `ParseForge.to_markdown()` → Markdown string
- Markdown saved to file
- `process_document()` reads markdown file

**Interface**:
- Markdown format with page markers (`--- Page X ---`)
- Identifier tags: `[TABLE]`, `[IMAGE]`, `[TOC]`, `[HEADER]`

### Chunking → Indexing

**Data Flow**:
- `process_document()` → `(List[Chunk], List[ParentChunk], Stats)`
- Chunks serialized to JSONL files
- `ingest_from_chunking_outputs()` reads JSONL files

**Interface**:
- JSONL format: One chunk per line
- Chunk fields: `chunk_id`, `doc_id`, `text_for_embedding`, `metadata`, etc.

### Indexing → Retrieval

**Data Flow**:
- `ingest_from_chunking_outputs()` → Pinecone vectors + Local chunk store
- `retrieve()` queries Pinecone → fetches chunks from store

**Interface**:
- Pinecone: Vector embeddings with metadata filters
- LocalChunkStore: JSON files keyed by `doc_id/chunk_id`

### Retrieval Components

**Reranker → Context Assembler**:
- `OpenAIReranker.rerank()` → `List[RerankResult]`
- `DefaultContextAssembler.assemble()` uses reranked results

**Context Assembler → Chunk Store**:
- Fetches chunks by `chunk_id` from `LocalChunkStore`
- Expands neighbors using `expand_neighbors()`

---

## API Reference

### ParseForge

```python
class ParseForge:
    def __init__(
        self,
        config: Optional[ParseForgeConfig] = None,
        progress_callback: Optional[Callable[[str, float, Optional[Dict]], None]] = None
    )
    
    def parse(
        self,
        file_path: str,
        strategy: StrategyEnum = StrategyEnum.AUTO,
        start_page: Optional[int] = None,
        end_page: Optional[int] = None
    ) -> Document
    
    def to_markdown(
        self,
        document: Document,
        generate_image_descriptions: bool = True
    ) -> str
```

### Chunking

```python
def process_document(
    file_path: str,
    config: ChunkConfig,
    doc_id: str
) -> Tuple[List[Chunk], List[ParentChunk], Dict[str, Any]]
```

### Retrieval

```python
def ingest_from_chunking_outputs(
    children_jsonl_path: str,
    parents_jsonl_path: str,
    doc_id: str,
    cfg: RetrievalConfig = None
) -> Dict[str, Any]

def retrieve(
    query: str,
    filters: Optional[Dict[str, Any]] = None,
    cfg: RetrievalConfig = None,
    all_chunks: Optional[List[Dict[str, Any]]] = None
) -> ContextPack
```

### RAGOrchestrator

```python
class RAGOrchestrator:
    def __init__(
        self,
        parsing_config: Optional[ParseForgeConfig] = None,
        chunking_config: Optional[ChunkConfig] = None,
        retrieval_config: Optional[RetrievalConfig] = None,
        progress_callback: Optional[callable] = None
    )
    
    def pipeline(
        self,
        file_path: str,
        doc_id: str,
        strategy: StrategyEnum = StrategyEnum.AUTO,
        generate_image_descriptions: bool = True,
        index_chunks: bool = True
    ) -> Dict[str, Any]
```

---

## License

MIT

---

## Contributing

This is a production-grade system. Contributions should maintain code quality, add tests, and update documentation.

---

## Support

For issues, questions, or contributions, please refer to the project repository.
