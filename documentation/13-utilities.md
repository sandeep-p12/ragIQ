# Utility Functions

## Overview

Utility functions provide common functionality used throughout the RAG IQ system.

## Environment Utilities

**Location**: `src/utils/env.py`

### Functions

**`load_env() -> Dict[str, str]`**:
- Loads all environment variables from `.env`
- Uses `python-dotenv`
- Returns dictionary of environment variables

**`get_openai_api_key() -> Optional[str]`**:
- Gets OpenAI API key from environment
- Returns None if not found

**`get_azure_openai_api_key() -> Optional[str]`**:
- Gets Azure OpenAI API key from environment
- Returns None if not found

**`get_azure_openai_endpoint() -> Optional[str]`**:
- Gets Azure OpenAI endpoint URL from environment
- Returns None if not found

**`get_azure_openai_api_version() -> str`**:
- Gets Azure OpenAI API version from environment
- Defaults to "2025-01-01-preview"

## Token Utilities

**Location**: `src/utils/tokens.py`

### Functions

**`count_tokens(text, model) -> int`**:
- Counts tokens in text using tiktoken
- Uses model-specific encoding (default: "gpt-3.5-turbo")
- Falls back to "cl100k_base" if model not found
- Returns token count

**`compute_structure_confidence(repair_records, page_content) -> float`**:
- Computes heuristic structure confidence
- Base confidence: 1.0
- Penalizes based on number of repairs (0.1 per repair, max 0.3)
- Penalizes by repair type: table_repair (-0.15), list_repair (-0.1), section_repair (-0.05)
- Boosts confidence if content has headings (+0.1)
- Returns value between 0.0 and 1.0
- Used for determining heading-based vs soft-section grouping

## ID Utilities

**Location**: `src/utils/ids.py`

### Functions

**`generate_chunk_id(content, metadata) -> str`**:
- Generates deterministic chunk ID using SHA256 hash
- Uses chunk content (raw_md_fragment) and metadata
- Creates stable string representation with sorted keys
- Returns 16-character hexadecimal hash
- Ensures same content + metadata = same ID

## I/O Utilities

**Location**: `src/utils/io.py`

### Functions

**`load_jsonl(file_path) -> List[Dict[str, Any]]`**:
- Loads JSONL file
- Reads line-by-line
- Parses each line as JSON
- Returns list of dictionaries
- Handles empty lines gracefully

**`save_jsonl(data, file_path) -> None`**:
- Saves list of dictionaries to JSONL file
- Writes each dictionary as one line
- Uses UTF-8 encoding
- Ensures ASCII-safe output

**`format_citation(chunk) -> str`**:
- Formats human-readable citation string
- Includes: Doc ID, Page span, Section label/header path
- Adds element-specific ranges (row_range, item_range) if available
- Format: `"Doc: {doc_id} | Page: {page} | Section: {section}"`

## Checkpoint Utilities

**Location**: `src/utils/checkpoint.py`

### Class: `Checkpoint`

**Purpose**: Checkpoint/resume mechanism for ParseForge parsing

**Initialization**:
- Takes `checkpoint_dir` Path
- Creates directory if doesn't exist

**Key Methods**:
- `save(document_path, last_batch, last_page, strategy, batches, checkpoint_id) -> str`:
  - Saves checkpoint to JSON file
  - Stores version, document path, batch/page progress, strategy, batch statuses
  - Returns checkpoint file path

- `load(checkpoint_id) -> Dict[str, Any]`:
  - Loads checkpoint JSON file
  - Returns checkpoint data dictionary
  - Raises `CheckpointError` if not found

- `list_checkpoints() -> List[str]`:
  - Lists all checkpoint IDs
  - Returns list of checkpoint file stems

- `delete(checkpoint_id) -> None`:
  - Deletes a checkpoint
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

## Bounding Box Utilities

**Location**: `src/utils/bbox.py`

### Functions

**Bounding box operations**:
- Normalization
- Intersection calculations
- IoU (Intersection over Union) calculations
- Coordinate transformations

## UI Utilities

**Location**: `src/utils/ui.py`

### Functions

**`display_error(error, details) -> None`**:
- Displays error in Streamlit UI
- Shows error message and optional details
- Displays exception traceback

**`format_progress_message(stage, progress) -> str`**:
- Formats progress message
- Converts progress (0-1) to percentage
- Returns formatted string

**`format_stage_output(stage, output_data) -> Dict[str, Any]`**:
- Formats output data for display
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

## Exception Utilities

**Location**: `src/utils/exceptions.py`

### Exception Classes

**`ParseForgeException`**: Base exception for all ParseForge errors

**`StrategyException(ParseForgeException)`**: Exception raised during strategy selection

**`OCRError(ParseForgeException)`**: Exception raised during OCR processing

**`LayoutError(ParseForgeException)`**: Exception raised during layout detection

**`TableError(ParseForgeException)`**: Exception raised during table processing

**`ParserError(ParseForgeException)`**: Exception raised during document parsing

**`CheckpointError(ParseForgeException)`**: Exception raised during checkpoint operations

**`ConfigurationError(ParseForgeException)`**: Exception raised due to configuration issues

## Memory Utilities

**Location**: `src/utils/memory.py`

### Functions

**Memory-related utilities**:
- Memory usage tracking
- Memory optimization helpers

## Usage Examples

### Environment Loading

```python
from src.utils.env import load_env, get_openai_api_key

env = load_env()
api_key = get_openai_api_key()
```

### Token Counting

```python
from src.utils.tokens import count_tokens

token_count = count_tokens("Hello, world!", model="gpt-3.5-turbo")
print(f"Token count: {token_count}")
```

### ID Generation

```python
from src.utils.ids import generate_chunk_id

chunk_id = generate_chunk_id("content", {"doc_id": "doc_1"})
print(f"Chunk ID: {chunk_id}")
```

### JSONL I/O

```python
from src.utils.io import load_jsonl, save_jsonl

# Load
chunks = load_jsonl("chunks.jsonl")

# Save
save_jsonl(chunks, "output.jsonl")
```

### Checkpoints

```python
from src.utils.checkpoint import Checkpoint
from pathlib import Path

checkpoint = Checkpoint(Path("checkpoints"))

# Save
checkpoint_id = checkpoint.save(
    document_path="doc.pdf",
    last_batch=5,
    last_page=50,
    strategy="AUTO",
    batches=[],
    checkpoint_id="checkpoint_1"
)

# Load
data = checkpoint.load("checkpoint_1")

# List
checkpoints = checkpoint.list_checkpoints()
```

## Next Steps

- **[API Reference](./14-api-reference.md)** - Utility API details
- **[Troubleshooting](./15-troubleshooting.md)** - Common utility issues

