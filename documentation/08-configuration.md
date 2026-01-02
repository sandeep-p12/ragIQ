# Configuration Guide

## Overview

RAG IQ uses a hierarchical configuration system with environment variables and configuration objects. All configuration is loaded from the `.env` file in the project root.

## Configuration Objects

### ParseForgeConfig

**Location**: `src/config/parsing.py`

**Created from environment variables with `PARSEFORGE_` prefix**

**Key Fields**:
- `device`: "cpu" | "cuda" | "mps" | "coreml"
- `batch_size`: Pages processed in parallel (default: 50)
- `page_threshold`: IoU threshold for page-level strategy (default: 0.6)
- `document_threshold`: Threshold for document-level strategy (default: 0.2)
- `finance_mode`: Enable finance mode (default: False)
- `llm_provider`: "openai" | "azure_openai" | "none"
- `llm_model`: Model name (default: "gpt-4o")
- `llm_azure_endpoint`: Azure OpenAI endpoint URL
- `llm_azure_deployment_name`: Azure OpenAI deployment name
- `llm_use_azure_ad`: Use Azure AD authentication (default: True)

**Example**:
```python
from src.config.parsing import ParseForgeConfig

config = ParseForgeConfig(
    device="cuda",
    batch_size=100,
    llm_provider="openai",
    llm_model="gpt-4o"
)
```

### ChunkConfig

**Location**: `src/config/chunking.py`

**Created directly (no environment variables)**

**Key Fields**:
- `prose_target_tokens`: Target size for prose chunks (default: 512)
- `prose_overlap_tokens`: Overlap between prose chunks (default: 50)
- `parent_heading_level`: Heading level for parent grouping (default: 2)
- `max_chunk_tokens_hard`: Hard limit for chunk tokens (default: 2048)
- `min_chunk_tokens`: Minimum target size (default: 100)

**Example**:
```python
from src.config.chunking import ChunkConfig

config = ChunkConfig(
    prose_target_tokens=512,
    prose_overlap_tokens=50,
    parent_heading_level=2,
    max_chunk_tokens_hard=2048
)
```

### RetrievalConfig

**Location**: `src/config/retrieval.py`

**Created from environment with `from_env()`**

**Key Fields**:
- `neighbor_same_page`: Same-page neighbors (default: 1)
- `neighbor_cross_page`: Cross-page neighbors (default: 2)
- `include_parents`: Include parent chunks (default: True)
- `final_max_tokens`: Token budget (default: 12000)

**Example**:
```python
from src.config.retrieval import RetrievalConfig

config = RetrievalConfig.from_env()
# Or with overrides:
config = RetrievalConfig.from_env(
    neighbor_same_page=2,
    final_max_tokens=15000
)
```

## Environment Variables

### ParseForge Configuration (Prefix: `PARSEFORGE_`)

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `PARSEFORGE_DEVICE` | str | `"cpu"` | Device: "cpu", "cuda", "mps", "coreml" |
| `PARSEFORGE_BATCH_SIZE` | int | `50` | Pages processed in parallel |
| `PARSEFORGE_PAGE_THRESHOLD` | float | `0.6` | IoU threshold for page-level strategy |
| `PARSEFORGE_DOCUMENT_THRESHOLD` | float | `0.2` | Threshold for document-level strategy |
| `PARSEFORGE_FINANCE_MODE` | bool | `False` | Enable finance mode (stricter thresholds) |
| `PARSEFORGE_LLM_PROVIDER` | str | `"openai"` | LLM provider: "openai", "azure_openai", or "none" |
| `PARSEFORGE_LLM_MODEL` | str | `"gpt-4o"` | LLM model name or Azure deployment name |
| `PARSEFORGE_LLM_API_KEY` | str | - | LLM API key (optional override) |
| `PARSEFORGE_LLM_MAX_TOKENS` | int | `1000` | Maximum tokens for LLM generation |
| `PARSEFORGE_LLM_AZURE_ENDPOINT` | str | - | Azure OpenAI endpoint URL |
| `PARSEFORGE_LLM_AZURE_API_VERSION` | str | `"2025-01-01-preview"` | Azure OpenAI API version |
| `PARSEFORGE_LLM_AZURE_DEPLOYMENT_NAME` | str | - | Azure OpenAI deployment name |
| `PARSEFORGE_LLM_USE_AZURE_AD` | bool | `True` | Use Azure AD authentication |

### OpenAI Configuration

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `OPENAI_API_KEY` | str | - | **Required** (if using OpenAI) OpenAI API key |

### Azure OpenAI Configuration

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `AZURE_OPENAI_ENDPOINT` | str | - | **Required** (if using Azure OpenAI) Azure OpenAI endpoint URL |
| `AZURE_OPENAI_API_KEY` | str | - | Azure OpenAI API key (optional if using Azure AD) |
| `AZURE_OPENAI_API_VERSION` | str | `"2025-01-01-preview"` | Azure OpenAI API version |
| `AZURE_OPENAI_DEPLOYMENT_NAME` | str | - | Azure OpenAI deployment name for LLM |
| `AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME` | str | - | Azure OpenAI deployment name for embeddings |
| `AZURE_OPENAI_USE_AZURE_AD` | bool | `True` | Use Azure AD authentication |
| `EMBEDDING_PROVIDER` | str | `"openai"` | Embedding provider: "openai" or "azure_openai" |

### Pinecone Configuration

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `PINECONE_API_KEY` | str | - | **Required** Pinecone API key |
| `PINECONE_INDEX_NAME` | str | `"hybrid-chunking"` | Pinecone index name |
| `PINECONE_NAMESPACE` | str | `"children"` | Pinecone namespace ("children" or "parents") |

## Example `.env` Files

### Using OpenAI

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

### Using Azure OpenAI

```bash
# Azure OpenAI (with Azure AD authentication - recommended)
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
AZURE_OPENAI_API_VERSION=2025-01-01-preview
AZURE_OPENAI_DEPLOYMENT_NAME=gpt-4o-deployment
AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME=text-embedding-3-small-deployment
AZURE_OPENAI_USE_AZURE_AD=true

# Pinecone
PINECONE_API_KEY=...
PINECONE_INDEX_NAME=hybrid-chunking
PINECONE_NAMESPACE=children

# ParseForge
PARSEFORGE_DEVICE=cpu
PARSEFORGE_BATCH_SIZE=50
PARSEFORGE_LLM_PROVIDER=azure_openai
PARSEFORGE_LLM_MODEL=gpt-4o
PARSEFORGE_LLM_AZURE_ENDPOINT=https://your-resource.openai.azure.com/
PARSEFORGE_LLM_AZURE_DEPLOYMENT_NAME=gpt-4o-deployment
PARSEFORGE_LLM_USE_AZURE_AD=true

# Embedding provider
EMBEDDING_PROVIDER=azure_openai
```

## Configuration Loading

### Environment Variable Loading

**Location**: `src/utils/env.py`

**Functions**:
- `load_env() -> Dict[str, str]`: Loads all environment variables from `.env`
- `get_openai_api_key() -> Optional[str]`: Gets OpenAI API key
- `get_azure_openai_endpoint() -> Optional[str]`: Gets Azure OpenAI endpoint
- `get_azure_openai_api_version() -> str`: Gets Azure OpenAI API version

### Pydantic Settings

Configuration objects use Pydantic Settings for automatic environment variable loading:
- Environment variable prefix matching
- Type validation
- Default values
- Case-insensitive matching

## Next Steps

- **[Azure OpenAI Setup](./09-azure-openai.md)** - Configure Azure OpenAI
- **[API Reference](./14-api-reference.md)** - Configuration API details

