# Overview

## What is RAG IQ?

RAG IQ is a complete end-to-end RAG (Retrieval-Augmented Generation) pipeline that transforms documents into searchable, retrievable knowledge bases. The system processes documents through four main stages:

1. **Parsing**: Extracts structured content from various document formats (PDF, DOCX, PPTX, XLSX, CSV, HTML, TXT, MD)
2. **Chunking**: Intelligently splits documents into semantically meaningful chunks with hierarchical parent-child relationships
3. **Indexing**: Embeds chunks and stores them in vector databases (Pinecone, Azure AI Search)
4. **Retrieval**: Retrieves relevant chunks using vector similarity, reranks with LLM, and assembles context packs

## Key Features

### Multi-format Document Parsing
- Supports 8+ document formats with intelligent strategy selection
- Automatic strategy detection (FAST vs HI_RES) based on document quality
- OCR-based extraction for scanned documents
- Vision LLM support for complex layouts
- Post-processing: reading order sorting, table merging, paragraph splitting

### Structure-First Chunking
- Preserves document structure while optimizing for token budgets
- Type-aware chunking (prose, lists, tables, images)
- Cross-page continuation detection and merging
- Markdown repair for malformed content

### Hybrid Hierarchy
- Creates parent-child chunk relationships for better context assembly
- Heading-based grouping for well-structured documents
- Soft-section grouping for documents with low structure confidence
- Configurable hierarchy levels

### LLM-Powered Reranking
- Uses GPT-4o to rerank retrieval candidates by relevance
- Strict JSON output for reliable parsing
- Relevance scoring (0-100 scale)
- Answerability detection

### Context Assembly
- Intelligently expands neighbors (same-page and cross-page)
- Token budget management
- Parent chunk inclusion
- Citation generation

### Unified Streamlit UI
- Complete pipeline visualization and control interface
- Real-time progress tracking
- Preview panels for all stages
- Export functionality
- Filtering and statistics

### Multi-Provider Support
- Supports both OpenAI and Azure OpenAI for LLM and embeddings
- Azure AD authentication using DefaultAzureCredential
- API key fallback for Azure OpenAI
- Mix and match providers (e.g., OpenAI LLM + Azure embeddings)

## System Capabilities

### Document Formats Supported

| Format | Extension | Parser | Notes |
|--------|-----------|--------|-------|
| PDF | `.pdf` | PDFParser | Multiple strategies (FAST, HI_RES, AUTO, LLM_FULL) |
| Word | `.docx` | DOCXParser | Preserves heading styles |
| PowerPoint | `.pptx` | PPTXParser | One page per slide |
| Excel | `.xlsx`, `.xls` | XLSXParser | One page per sheet |
| CSV | `.csv` | CSVParser | First row as headers |
| HTML | `.html`, `.htm` | HTMLParser | Extracts from common tags |
| Text | `.txt` | TXTParser | Line-by-line processing |
| Markdown | `.md` | MDParser | Heading detection |

### Parsing Strategies

1. **FAST**: Native text extraction using pypdfium2
   - Fastest option
   - Works for text-based PDFs
   - No OCR required

2. **HI_RES**: OCR-based extraction using Doctr
   - For scanned documents
   - Layout detection with YOLO
   - Higher accuracy for images

3. **AUTO**: Automatic strategy selection
   - Analyzes each page
   - Chooses FAST or HI_RES per page
   - Optimizes for speed and accuracy

4. **LLM_FULL**: Full document parsing using LLM vision
   - Uses GPT-4o vision
   - Best for complex layouts
   - Most accurate but slowest

### Chunking Features

- **Prose Chunking**: Sentence-aware splitting with windowing
- **List Chunking**: Preserves list structure, splits by items
- **Table Chunking**: Row-based splitting with overlap
- **Image Chunking**: Handles image blocks with descriptions
- **Token Budget Refinement**: Splits oversized, merges tiny chunks
- **Cross-Page Merging**: Detects and merges continuations

### Retrieval Features

- **Vector Search**: Pinecone integration for similarity search
- **LLM Reranking**: Relevance scoring with GPT-4o
- **Neighbor Expansion**: Includes context from adjacent chunks
- **Parent Inclusion**: Adds parent chunks for broader context
- **Token Budgeting**: Manages context size within limits
- **Citation Generation**: Automatic citation formatting

## Use Cases

- **Document Q&A**: Answer questions from large document collections
- **Knowledge Base Search**: Search across technical documentation
- **Research Assistance**: Extract and retrieve information from papers
- **Content Analysis**: Analyze and understand document structure
- **Information Extraction**: Extract structured data from unstructured documents

## Next Steps

- **[Setup Guide](./02-setup.md)** - Get started with installation
- **[Architecture](./03-architecture.md)** - Understand the system design
- **[Parsing](./04-parsing.md)** - Learn about document parsing
- **[Chunking](./05-chunking.md)** - Understand intelligent chunking
- **[Retrieval](./07-retrieval.md)** - Learn about retrieval pipeline

