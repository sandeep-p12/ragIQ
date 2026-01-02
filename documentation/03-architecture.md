# System Architecture

## High-Level Architecture

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

## Component Layers

### 1. UI Layer
**Purpose**: Streamlit application providing interactive interface

**Components**:
- `streamlit_app.py`: Main UI application
- Tabbed interface for Parse, Chunk, Index, Retrieve, Pipeline
- Real-time progress tracking
- Configuration panels
- Preview and export functionality

### 2. Orchestration Layer
**Purpose**: Coordinates parsing, chunking, and retrieval

**Components**:
- `RAGOrchestrator`: Unified orchestrator class
- Progress callbacks
- Pipeline coordination
- Error handling

### 3. Pipeline Layer
**Purpose**: Domain-specific pipelines

**Components**:
- **Parsing Pipeline**: `pipelines/parsing/`
  - `parseforge.py`: Main orchestrator
  - `parsers/`: Format-specific parsers
  - `processing/`: Post-processing (reading order, table merging, etc.)
  
- **Chunking Pipeline**: `pipelines/chunking/`
  - `chunking.py`: Main orchestration
  - `chunkers.py`: Structure-first chunking and refinement
  - `hierarchy.py`: Parent-child grouping
  - `repair.py`: Markdown repair
  
- **Retrieval Pipeline**: `pipelines/retrieval/`
  - `retrieval.py`: Main retrieval functions
  - `reranker.py`: LLM reranking
  - `context_assembler.py`: Context assembly

### 4. Provider Layer
**Purpose**: External service integrations

**Components**:
- **LLM Provider**: `providers/llm/openai_llm.py`
  - OpenAI and Azure OpenAI support
  - Vision capabilities
  
- **Embedding Provider**: `providers/embedding/openai_embedding.py`
  - OpenAI and Azure OpenAI embeddings
  - Batch processing and caching
  
- **OCR Provider**: `providers/ocr/doctr.py`
  - Doctr OCR for text detection
  
- **Layout Provider**: `providers/layout/yolo.py`
  - YOLO layout detection

### 5. Storage Layer
**Purpose**: Vector stores and chunk stores

**Components**:
- **Vector Store**: `storage/vector/`
  - `pinecone.py`: Pinecone integration
  - `azure_ai_search.py`: Azure AI Search (placeholder)
  
- **Chunk Store**: `storage/chunk/`
  - `local.py`: Local file-based storage
  - `azure_blob.py`: Azure Blob Storage (placeholder)

### 6. Schema Layer
**Purpose**: Data models and type definitions

**Components**:
- `schema/document.py`: Document, Page, Block models
- `schema/chunk.py`: Chunk, ParentChunk, Element types

### 7. Config Layer
**Purpose**: Centralized configuration management

**Components**:
- `config/parsing.py`: ParseForge configuration
- `config/chunking.py`: Chunking configuration
- `config/retrieval.py`: Retrieval configuration
- `config/parsing_strategies.py`: Strategy selection logic
- `config/prompts.py`: LLM prompts

## Data Flow

### End-to-End Flow

```
Document File (PDF/DOCX/etc.)
    │
    ▼
[ParseForge.parse()]
    │
    ├─► Format Detection
    ├─► Strategy Selection
    ├─► Page Processing
    ├─► Post-Processing
    │
    ▼
Document Object (Pydantic Model)
    │
    ▼
[ParseForge.to_markdown()]
    │
    ▼
Markdown String
    │
    ▼
[process_document()]
    │
    ├─► Markdown Repair
    ├─► Page Block Splitting
    ├─► LlamaIndex Parsing
    ├─► Element Extraction
    ├─► Structure-First Chunking
    ├─► Token-Budget Refinement
    ├─► Hierarchy Building
    │
    ▼
(List[Chunk], List[ParentChunk], Stats)
    │
    ▼
[ingest_from_chunking_outputs()]
    │
    ├─► Local Chunk Storage
    ├─► Text Embedding
    ├─► Vector Record Creation
    └─► Pinecone Upsert
    │
    ▼
Indexed Vector Database
    │
    ▼
[retrieve()]
    │
    ├─► Query Embedding
    ├─► Pinecone Query
    ├─► Candidate Text Fetching
    ├─► LLM Reranking
    ├─► Neighbor Expansion
    ├─► Context Assembly
    └─► Token Budgeting
    │
    ▼
ContextPack (selected_chunks, citations, trace)
```

## Design Patterns

### Protocol-Based Interfaces
- Uses Python Protocols for duck typing
- Enables easy swapping of implementations
- Defined in `core/interfaces.py`

### Configuration Management
- Pydantic Settings for type-safe configuration
- Environment variable loading
- Hierarchical configuration objects

### Progress Callbacks
- Optional progress tracking throughout pipelines
- Stage-based progress updates
- Output data for UI display

## Project Structure

```
ragIQ/
├── src/
│   ├── config/          # Configuration management
│   ├── core/            # Core interfaces and dataclasses
│   ├── pipelines/       # Main pipelines
│   │   ├── parsing/     # Document parsing
│   │   ├── chunking/    # Document chunking
│   │   └── retrieval/   # Retrieval pipeline
│   ├── providers/       # External service providers
│   ├── storage/         # Storage backends
│   ├── schema/          # Data models
│   ├── formatters/      # Output formatters
│   └── utils/           # Utility functions
├── streamlit_app.py     # Streamlit UI
└── pyproject.toml       # Project configuration
```

## Next Steps

- **[Parsing Pipeline](./04-parsing.md)** - Detailed parsing architecture
- **[Chunking Pipeline](./05-chunking.md)** - Chunking architecture
- **[Retrieval Pipeline](./07-retrieval.md)** - Retrieval architecture

