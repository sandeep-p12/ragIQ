# Document Parsing Pipeline

## Overview

The parsing pipeline extracts structured content from various document formats and converts them into a unified `Document` model. It supports multiple parsing strategies optimized for different document types and quality levels.

## ParseForge Orchestrator

### Class: `ParseForge`

**Location**: `src/pipelines/parsing/parseforge.py`

**Initialization**:
```python
from src.pipelines.parsing.parseforge import ParseForge
from src.config.parsing import ParseForgeConfig

config = ParseForgeConfig()
parser = ParseForge(config, progress_callback=callback)
```

**Key Methods**:
- `parse(file_path, strategy, start_page, end_page) -> Document`: Main parsing method
- `to_markdown(document, generate_image_descriptions) -> str`: Converts Document to markdown

## Supported Formats

| Format | Extension | Parser | Implementation |
|--------|-----------|--------|----------------|
| PDF | `.pdf` | PDFParser | `parsers/pdf.py` |
| Word | `.docx` | parse_docx | `parsers/docx.py` |
| PowerPoint | `.pptx` | parse_pptx | `parsers/pptx.py` |
| Excel | `.xlsx`, `.xls` | parse_xlsx | `parsers/xlsx.py` |
| CSV | `.csv` | parse_csv | `parsers/csv.py` |
| HTML | `.html`, `.htm` | parse_html | `parsers/html_txt_md.py` |
| Text | `.txt` | parse_txt_md | `parsers/html_txt_md.py` |
| Markdown | `.md` | parse_txt_md | `parsers/html_txt_md.py` |

## Parsing Strategies

### StrategyEnum

**Location**: `src/config/parsing_strategies.py`

**Values**:
- `FAST`: Native text extraction (pypdfium2)
- `HI_RES`: OCR-based extraction (Doctr)
- `AUTO`: Automatic selection per page
- `LLM_FULL`: Full document parsing using LLM vision

### FAST Strategy

**Use Case**: Text-based PDFs with good quality

**Process**:
1. Native text extraction using pypdfium2
2. Block processing
3. Reading order sorting

**Advantages**:
- Fastest option
- No OCR required
- Preserves original text formatting

### HI_RES Strategy

**Use Case**: Scanned documents, images, poor quality PDFs

**Process**:
1. OCR detection (Doctr)
2. Layout detection (YOLO)
3. Block assembly
4. Reading order sorting

**Advantages**:
- Works with scanned documents
- Layout-aware extraction
- Higher accuracy for images

### AUTO Strategy

**Use Case**: Mixed quality documents

**Process**:
1. Runs OCR detection on all pages
2. Determines per-page strategy based on IoU threshold
3. Determines global strategy
4. Processes pages accordingly

**Logic**:
- Compares native text extraction vs OCR detection
- Uses IoU (Intersection over Union) to measure overlap
- If IoU < threshold → use HI_RES, else FAST
- Document-level: If > threshold% pages need HI_RES → use HI_RES globally

### LLM_FULL Strategy

**Use Case**: Complex layouts, best accuracy needed

**Process**:
1. Vision LLM processes entire document
2. Generates markdown with layout awareness
3. Extracts tables, images, text

**Advantages**:
- Best accuracy
- Handles complex layouts
- Understands document structure

**Disadvantages**:
- Slowest option
- Requires LLM API access
- Higher cost

## PDF Parser

### Class: `PDFParser`

**Location**: `src/pipelines/parsing/parsers/pdf.py`

**Initialization**:
- Initializes `DoctrOCR`, `YOLOLayoutDetector`, `ImageVisionLLMFormatter`
- Handles missing dependencies gracefully

**Key Methods**:
- `parse(file_path, strategy, start_page, end_page) -> Document`
- `_parse_with_llm_full()`: Full document parsing using LLM vision
- `_process_page_batch()`: Processes a batch of pages
- `_process_page_fast()`: FAST strategy (native extraction)
- `_process_page_hi_res()`: HI_RES strategy (OCR + layout)

## Post-Processing

### Reading Order Sorting

**Location**: `src/pipelines/parsing/processing/reading_order.py`

**Algorithm**: XY-Cut algorithm

**Function**: `sort_blocks_by_reading_order(blocks, page_width, page_height)`

**Features**:
- Handles multi-column layouts
- Preserves reading order (top-to-bottom, left-to-right)
- Recursive XY-Cut algorithm

### Table Merging

**Location**: `src/pipelines/parsing/processing/table_merger.py`

**Function**: `merge_cross_page_tables(tables)`

**Features**:
- Identifies continuation tables across pages
- Merges cells and rows
- Preserves headers
- Header detection

### Paragraph Splitting

**Location**: `src/pipelines/parsing/processing/para_split.py`

**Function**: `split_paragraphs(blocks)`

**Features**:
- Splits long paragraphs into smaller blocks
- Preserves paragraph structure
- Handles lists and indexes separately
- List and index detection

### Table Extraction

**Location**: `src/pipelines/parsing/processing/table_extractor.py`

**Function**: `extract_table(block, page_width, page_height, use_vision_llm)`

**Features**:
- Extracts table from TextBlock or ImageBlock
- Uses OCR text or vision LLM for image-based tables
- Builds table grid from text/OCR
- Extracts cells and headers

## Format-Specific Parsers

### DOCX Parser

**Location**: `src/pipelines/parsing/parsers/docx.py`

**Implementation**:
- Uses `python-docx` library
- Determines block type based on paragraph style
- Styles containing "heading" or "title" → `TitleBlock`
- Other paragraphs → `TextBlock`
- Creates single `Page` with all blocks

### PPTX Parser

**Location**: `src/pipelines/parsing/parsers/pptx.py`

**Implementation**:
- Uses `python-pptx` library
- One page per slide
- Detects title placeholders
- Extracts text from shapes

### XLSX Parser

**Location**: `src/pipelines/parsing/parsers/xlsx.py`

**Implementation**:
- Uses `openpyxl` library
- One page per sheet
- Converts each sheet to a `TableBlock`
- Extracts all rows and cells

### CSV Parser

**Location**: `src/pipelines/parsing/parsers/csv.py`

**Implementation**:
- Uses `pandas` library
- First row: column headers
- Subsequent rows: data rows
- Creates single `Page` with `TableBlock`

### HTML/TXT/MD Parser

**Location**: `src/pipelines/parsing/parsers/html_txt_md.py`

**HTML Implementation**:
- Uses BeautifulSoup
- Extracts from common elements: `p`, `h1-h6`, `div`
- Tags starting with `h` → `TitleBlock`
- Other tags → `TextBlock`

**TXT/MD Implementation**:
- Reads file line-by-line
- Detects markdown headings: lines starting with `#`
- Groups consecutive non-empty lines into paragraphs

## Markdown Generation

### Function: `to_markdown(document, generate_image_descriptions)`

**Location**: `src/pipelines/parsing/parseforge.py`

**Process**:
1. Block-to-Markdown conversion
2. Table formatting (LLM) - optional
3. Image description generation (LLM Vision) - optional

**Output Format**:
- Page markers: `--- Page X ---`
- Identifier tags: `[TABLE]`, `[IMAGE]`, `[TOC]`, `[HEADER]`
- Standard markdown formatting

## Formatters

### Markdown Formatter

**Location**: `src/formatters/markdown.py`

**Function**: `blocks_to_markdown(blocks, image_formatter, image_descriptions)`

**Features**:
- Formats title blocks with heading levels
- Formats text blocks with whitespace normalization
- Formats list blocks (ordered/unordered)
- Formats table blocks with cell sanitization
- Formats image blocks with descriptions
- Formats code blocks with language tags

### Table LLM Formatter

**Location**: `src/formatters/table.py`

**Class**: `TableLLMFormatter`

**Purpose**: Formats table blocks using LLM

**Process**:
1. Extracts table text from TableBlock
2. Calls LLM with formatting prompt
3. Extracts markdown table from response
4. Stores markdown in HTML with markers

### Image Vision LLM Formatter

**Location**: `src/formatters/image.py`

**Class**: `ImageVisionLLMFormatter`

**Purpose**: Generates image descriptions using vision LLM

**Methods**:
- `describe_image(image_block) -> str`: Generates description for a single image
- `describe_images_batch(image_blocks) -> List[str]`: Batch processing
- `process_page_with_images()`: Processes whole page with OCR + image descriptions

## Checkpoint System

**Location**: `src/utils/checkpoint.py`

**Class**: `Checkpoint`

**Purpose**: Resume parsing from checkpoints

**Features**:
- Saves progress after each batch
- Stores document path, batch/page progress, strategy
- Auto-resume capability
- Checkpoint management (list, delete)

## Configuration

See [Configuration Guide](./08-configuration.md) for parsing configuration options.

## Next Steps

- **[Chunking Pipeline](./05-chunking.md)** - Process parsed documents into chunks
- **[Configuration](./08-configuration.md)** - Configure parsing settings
- **[API Reference](./14-api-reference.md)** - Parsing API details

