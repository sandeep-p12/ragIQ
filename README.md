# RAG IQ

**Production-grade RAG (Retrieval-Augmented Generation) system** with comprehensive document parsing, intelligent chunking, vector embedding, indexing, and retrieval capabilities.

## Quick Start

1. **Install Dependencies**
   ```bash
   uv sync
   # or
   pip install -e .
   ```

2. **Configure Environment**
   ```bash
   # Create .env file with your API keys
   OPENAI_API_KEY=sk-...
   PINECONE_API_KEY=...
   ```

3. **Run Streamlit UI**
   ```bash
   streamlit run streamlit_app.py
   ```

## Documentation

Comprehensive documentation is available in the [`documentation/`](./documentation/) folder:

### Getting Started
- **[01 - Overview](./documentation/01-overview.md)** - System overview, key features, and capabilities
- **[02 - Setup](./documentation/02-setup.md)** - Installation, prerequisites, and initial configuration
- **[03 - Architecture](./documentation/03-architecture.md)** - System architecture and component layers

### Core Components
- **[04 - Parsing](./documentation/04-parsing.md)** - Document parsing pipeline, strategies, and format support
- **[05 - Chunking](./documentation/05-chunking.md)** - Intelligent chunking with hierarchical relationships
- **[06 - Embedding & Indexing](./documentation/06-embedding-indexing.md)** - Vector embedding and indexing to vector stores
- **[07 - Retrieval](./documentation/07-retrieval.md)** - Query processing, reranking, and context assembly

### Configuration & Integration
- **[08 - Configuration](./documentation/08-configuration.md)** - Complete configuration guide and environment variables
- **[09 - Azure OpenAI](./documentation/09-azure-openai.md)** - Azure OpenAI setup and authentication
- **[10 - Providers](./documentation/10-providers.md)** - LLM, OCR, and Layout detection providers
- **[11 - Storage](./documentation/11-storage.md)** - Vector stores and chunk storage backends

### Reference
- **[12 - Schemas](./documentation/12-schemas.md)** - Data models and schema definitions
- **[13 - Utilities](./documentation/13-utilities.md)** - Utility functions and helpers
- **[14 - API Reference](./documentation/14-api-reference.md)** - Complete API documentation
- **[15 - Troubleshooting](./documentation/15-troubleshooting.md)** - Common issues and solutions

## Key Features

- **Multi-format Document Parsing**: Supports 8+ document formats (PDF, DOCX, PPTX, XLSX, CSV, HTML, TXT, MD)
- **Structure-First Chunking**: Preserves document structure while optimizing for token budgets
- **Hybrid Hierarchy**: Creates parent-child chunk relationships for better context assembly
- **LLM-Powered Reranking**: Uses GPT-4o to rerank retrieval candidates by relevance
- **Context Assembly**: Intelligently expands neighbors and assembles context packs within token budgets
- **Unified Streamlit UI**: Complete pipeline visualization and control interface
- **Multi-Provider Support**: Supports both OpenAI and Azure OpenAI for LLM and embeddings
- **Azure AD Authentication**: Secure authentication using Azure AD (DefaultAzureCredential) with API key fallback

## System Overview

RAG IQ processes documents through four main stages:

1. **Parsing**: Extracts structured content from various document formats
2. **Chunking**: Intelligently splits documents into semantically meaningful chunks
3. **Indexing**: Embeds chunks and stores them in vector databases
4. **Retrieval**: Retrieves relevant chunks using vector similarity, reranks with LLM, and assembles context packs

## Project Structure

```
ragIQ/
├── documentation/          # Comprehensive documentation
├── src/                     # Source code
│   ├── config/             # Configuration management
│   ├── core/               # Core interfaces and dataclasses
│   ├── pipelines/          # Main pipelines (parsing, chunking, retrieval)
│   ├── providers/          # External service providers
│   ├── storage/           # Storage backends
│   ├── schema/            # Data models
│   └── utils/             # Utility functions
├── streamlit_app.py        # Streamlit UI
└── pyproject.toml          # Project configuration
```

## Requirements

- Python >= 3.10
- OpenAI API key (or Azure OpenAI credentials)
- Pinecone API key (for vector storage)

## License

MIT

## Support

For detailed documentation, see the [`documentation/`](./documentation/) folder. For issues, questions, or contributions, please refer to the project repository.
