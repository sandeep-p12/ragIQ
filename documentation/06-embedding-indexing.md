# Embedding & Indexing

## Overview

The embedding and indexing pipeline converts chunks into vector embeddings and stores them in vector databases for efficient similarity search.

## Embedding Provider

**Location**: `src/providers/embedding/openai_embedding.py`

### Class: `OpenAIEmbeddingProvider`

**Supports**:
- OpenAI embeddings
- Azure OpenAI embeddings

**Initialization**:
```python
from src.config.retrieval import EmbeddingConfig
from src.providers.embedding.openai_embedding import OpenAIEmbeddingProvider

config = EmbeddingConfig.from_env(provider="openai")
provider = OpenAIEmbeddingProvider(config)
```

**Key Methods**:
- `embed_texts(texts) -> List[List[float]]`: Embeds batch of texts
- `embed_query(query) -> List[float]`: Embeds single query

**Features**:
- Caching by model + text
- Batch processing (`batch_size` from config)
- Retry logic with exponential backoff
- Rate limit handling
- Azure AD authentication support

## Indexing Function

**Location**: `src/pipelines/retrieval/retrieval.py`

### Function: `ingest_from_chunking_outputs(children_jsonl_path, parents_jsonl_path, doc_id, cfg)`

**Process**:
1. Loads JSONL files (children and parents)
2. Stores chunks locally (`LocalChunkStore`)
3. Embeds children texts (`OpenAIEmbeddingProvider`)
4. Creates `VectorRecord`s with metadata
5. Upserts to Pinecone (children and optionally parents)
6. Returns ingestion stats

**Returns**: Dictionary with ingestion statistics

## Vector Record

**Location**: `src/core/dataclasses.py`

### Class: `VectorRecord`

**Fields**:
- `id`: Vector ID (chunk_id)
- `values`: List[float] - Embedding vector
- `metadata`: Dict - Chunk metadata

**Metadata Includes**:
- `doc_id`: Document ID
- `chunk_id`: Chunk ID
- `page_span`: Page span tuple
- `section_label`: Section label
- `element_type`: Element type
- `parent_id`: Parent chunk ID (if applicable)

## Vector Store

**Location**: `src/storage/vector/pinecone.py`

### Class: `PineconeVectorStore`

**Initialization**:
- Creates index if doesn't exist (serverless or pod-based)
- Determines dimension from embedding model

**Key Methods**:
- `upsert(vectors, namespace)`: Upserts vectors to Pinecone
  - Batches vectors (100 at a time)
  - Sanitizes metadata (JSON-serializable)
  
- `query(vector, top_k, filters, namespace) -> List[Candidate]`:
  - Builds Pinecone filter from metadata filters
  - Returns `Candidate` objects with scores

**Features**:
- Automatic index creation
- Metadata sanitization
- Batch upserting
- Filter support

## Chunk Store

**Location**: `src/storage/chunk/local.py`

### Class: `LocalChunkStore`

**Purpose**: Local file-based chunk storage

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

**Key Methods**:
- `put_chunks(children, parents)`: Stores chunks to disk
  - Creates directory structure
  - Atomic writes (temp file + rename)
  
- `get_chunk(doc_id, chunk_id, is_parent) -> Dict`: Gets single chunk
- `get_chunks_bulk(keys) -> List[Dict]`: Gets multiple chunks

## Configuration

**Location**: `src/config/retrieval.py`

### EmbeddingConfig

**Fields**:
- `provider`: "openai" or "azure_openai" (default: "openai")
- `model`: Model name (default: "text-embedding-3-small")
- `batch_size`: Batch size (default: 100)
- `max_retries`: Max retries (default: 3)
- `azure_endpoint`: Azure OpenAI endpoint URL (required for Azure)
- `azure_deployment_name`: Azure OpenAI deployment name (required for Azure)
- `azure_api_version`: Azure OpenAI API version (default: "2025-01-01-preview")
- `use_azure_ad`: Use Azure AD authentication (default: True)

### PineconeConfig

**Fields**:
- `api_key_env`: Environment variable name (default: "PINECONE_API_KEY")
- `index_name_env`: Environment variable name (default: "PINECONE_INDEX_NAME")
- `namespace`: Namespace ("children" | "parents", default: "children")
- `top_k_dense`: Number of dense vectors to retrieve (default: 300)

## Usage Example

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

print(f"Indexed {stats['chunks_indexed']} chunks")
print(f"Created {stats['vectors_created']} vectors")
```

## Environment Variables

**Required**:
- `PINECONE_API_KEY`: Pinecone API key
- `OPENAI_API_KEY` or `AZURE_OPENAI_ENDPOINT`: For embeddings

**Optional**:
- `PINECONE_INDEX_NAME`: Index name (default: "hybrid-chunking")
- `PINECONE_NAMESPACE`: Namespace (default: "children")
- `EMBEDDING_PROVIDER`: "openai" or "azure_openai"

## Indexing Statistics

The indexing function returns statistics including:
- `chunks_indexed`: Number of chunks indexed
- `vectors_created`: Number of vectors created
- `time_seconds`: Time taken
- `embedding_stats`: Embedding statistics
- `pinecone_stats`: Pinecone statistics

## Next Steps

- **[Retrieval](./07-retrieval.md)** - Query indexed chunks
- **[Azure OpenAI](./09-azure-openai.md)** - Configure Azure OpenAI embeddings
- **[Storage](./11-storage.md)** - Storage backend details

