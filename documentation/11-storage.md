# Storage Backends

## Overview

RAG IQ uses two types of storage: vector stores for embeddings and chunk stores for chunk data.

## Vector Stores

### Pinecone Vector Store

**Location**: `src/storage/vector/pinecone.py`

**Class**: `PineconeVectorStore`

**Initialization**:
```python
from src.config.retrieval import PineconeConfig
from src.storage.vector.pinecone import PineconeVectorStore

config = PineconeConfig.from_env()
store = PineconeVectorStore(config)
```

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
- Namespace support (children/parents)

**Configuration**:
- `api_key`: Pinecone API key (from `PINECONE_API_KEY`)
- `index_name`: Index name (from `PINECONE_INDEX_NAME`, default: "hybrid-chunking")
- `namespace`: Namespace ("children" | "parents", default: "children")
- `top_k_dense`: Number of dense vectors to retrieve (default: 300)

### Azure AI Search (Placeholder)

**Location**: `src/storage/vector/azure_ai_search.py`

**Class**: `AzureAISearchStore`

**Status**: Not yet implemented - placeholder for future Azure AI Search integration.

**Interface**: Matches `VectorStore` protocol for easy migration:
- `upsert(vectors, namespace) -> None`
- `query(vector, top_k, filters, namespace) -> List[Candidate]`

**Note**: Currently raises `NotImplementedError`. Use `PineconeVectorStore` for now.

## Chunk Stores

### Local Chunk Store

**Location**: `src/storage/chunk/local.py`

**Class**: `LocalChunkStore`

**Purpose**: Local file-based chunk storage

**Initialization**:
```python
from src.storage.chunk.local import LocalChunkStore

store = LocalChunkStore()
```

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

**Chunk JSON Format**:
```json
{
  "chunk_id": "...",
  "doc_id": "...",
  "page_span": [1, 2],
  "page_nos": [1, 2],
  "header_path": "H1 > H2",
  "section_label": "...",
  "element_type": "paragraph",
  "raw_md_fragment": "...",
  "text_for_embedding": "...",
  "metadata": {...},
  "parent_id": "...",
  "token_count": 150,
  "node_id": "...",
  "line_start": 10,
  "line_end": 25
}
```

### Azure Blob Storage (Placeholder)

**Location**: `src/storage/chunk/azure_blob.py`

**Class**: `AzureBlobStore`

**Status**: Not yet implemented - placeholder for future Azure Blob Storage integration.

**Interface**: Matches `ChunkStore` protocol for easy migration:
- `put_chunks(children, parents) -> None`
- `get_chunk(doc_id, chunk_id, is_parent) -> Optional[Dict]`
- `get_chunks_bulk(keys) -> List[Optional[Dict]]`

**Note**: Currently raises `NotImplementedError`. Use `LocalChunkStore` for now.

## Storage Interfaces

**Location**: `src/core/interfaces.py`

### VectorStore Protocol

```python
class VectorStore(Protocol):
    def upsert(self, vectors: List[VectorRecord], namespace: str) -> None: ...
    def query(
        self,
        vector: List[float],
        top_k: int,
        filters: Optional[Dict[str, Any]] = None,
        namespace: str = "children"
    ) -> List[Candidate]: ...
```

### ChunkStore Protocol

```python
class ChunkStore(Protocol):
    def put_chunks(
        self,
        children: List[Dict[str, Any]],
        parents: List[Dict[str, Any]]
    ) -> None: ...
    
    def get_chunk(
        self,
        doc_id: str,
        chunk_id: str,
        is_parent: bool = False
    ) -> Optional[Dict[str, Any]]: ...
    
    def get_chunks_bulk(
        self,
        keys: List[Tuple[str, str, bool]]
    ) -> List[Optional[Dict[str, Any]]]: ...
```

## Usage Examples

### Vector Store

```python
from src.storage.vector.pinecone import PineconeVectorStore
from src.core.dataclasses import VectorRecord

store = PineconeVectorStore(config)

# Upsert vectors
vectors = [
    VectorRecord(
        id="chunk_1",
        values=[0.1, 0.2, ...],
        metadata={"doc_id": "doc_1", "page_span": [1, 2]}
    )
]
store.upsert(vectors, namespace="children")

# Query
results = store.query(
    vector=[0.1, 0.2, ...],
    top_k=10,
    filters={"doc_id": "doc_1"},
    namespace="children"
)
```

### Chunk Store

```python
from src.storage.chunk.local import LocalChunkStore

store = LocalChunkStore()

# Store chunks
store.put_chunks(children, parents)

# Get chunk
chunk = store.get_chunk("doc_1", "chunk_1", is_parent=False)

# Get multiple chunks
keys = [("doc_1", "chunk_1", False), ("doc_1", "chunk_2", False)]
chunks = store.get_chunks_bulk(keys)
```

## Performance Considerations

### Vector Store
- Batch upserting (100 vectors at a time)
- Metadata size limits (Pinecone has limits)
- Index dimension must match embedding dimension

### Chunk Store
- Atomic writes prevent corruption
- File-based storage is fast for local access
- Consider cloud storage for distributed systems

## Next Steps

- **[Configuration](./08-configuration.md)** - Configure storage settings
- **[Embedding & Indexing](./06-embedding-indexing.md)** - Index chunks to storage
- **[Retrieval](./07-retrieval.md)** - Retrieve from storage

