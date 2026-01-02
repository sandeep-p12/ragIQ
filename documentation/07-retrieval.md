# Retrieval Pipeline

## Overview

The retrieval pipeline retrieves relevant chunks using vector similarity, reranks with LLM, and assembles context packs within token budgets.

## Main Retrieval Function

**Location**: `src/pipelines/retrieval/retrieval.py`

### Function: `retrieve(query, filters, cfg, all_chunks, llm_config) -> ContextPack`

**Process**:
1. Embeds query using `OpenAIEmbeddingProvider`
2. Queries Pinecone (`top_k_dense` candidates)
3. Fetches candidate text snippets from chunk store
4. Reranks candidates with LLM
5. Assembles context pack with neighbor expansion
6. Returns `ContextPack`

## Reranking

**Location**: `src/pipelines/retrieval/reranker.py`

### Class: `OpenAIReranker`

**Purpose**: Reranks retrieval candidates using LLM

**Method**: `rerank(query, candidates) -> List[RerankResult]`

**Process**:
1. Builds prompt with query and candidates
2. Calls LLM with JSON response format
3. Parses and validates response
4. Returns sorted results (by relevance_score descending)

**Scoring Rubric**:
- **100**: Directly answers query with explicit evidence
- **70**: Highly relevant but partial answer
- **40**: Tangential context
- **0**: Unrelated

**Features**:
- Strict JSON output
- Retry logic with exponential backoff
- Caching by query + candidate IDs
- Candidate text trimming (`max_text_chars_per_candidate`)

**RerankResult Fields**:
- `chunk_id`: Chunk ID
- `relevance_score`: Score (0-100)
- `answerability`: Boolean
- `key_evidence`: Key evidence string

## Context Assembly

**Location**: `src/pipelines/retrieval/context_assembler.py`

### Class: `DefaultContextAssembler`

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

## ContextPack

**Location**: `src/core/dataclasses.py`

### Class: `ContextPack`

**Fields**:
- `query`: Original query string
- `selected_chunks`: List of selected chunk dictionaries
- `citations`: List of citation strings
- `trace`: Dictionary with retrieval trace
- `context_pack`: Formatted context string

**Trace Includes**:
- `pinecone_candidates`: Initial candidates from Pinecone
- `rerank_results`: Reranked results
- `final_token_count`: Final token count
- `expansion_stats`: Neighbor expansion statistics

## Configuration

**Location**: `src/config/retrieval.py`

### RetrievalConfig

**Fields**:
- `neighbor_same_page`: Same-page neighbors (default: 1)
- `neighbor_cross_page`: Cross-page neighbors (default: 2)
- `include_parents`: Include parent chunks (default: True)
- `final_max_tokens`: Token budget for final context (default: 12000)
- `min_primary_hits_to_keep`: Minimum primary chunks to keep (default: 3)

### RerankConfig

**Fields**:
- `model`: Model name (default: "gpt-4o")
- `max_candidates_to_rerank`: Max candidates to rerank (default: 50)
- `return_top_n`: Final number of chunks (default: 15)
- `max_text_chars_per_candidate`: Max chars per candidate (default: 1200)
- `strict_json_output`: Enforce strict JSON (default: True)

## Usage Example

```python
from src.config.retrieval import RetrievalConfig
from src.pipelines.retrieval.retrieval import retrieve

config = RetrievalConfig.from_env()

result = retrieve(
    "What is the main topic?",
    filters={"doc_id": "doc_1"},
    cfg=config,
    all_chunks=[],
    llm_config=None
)

print(f"Retrieved {len(result.selected_chunks)} chunks")
print(f"Citations: {result.citations}")
print(f"Context pack: {result.context_pack}")
```

## Retrieval Flow

```
Query String
    │
    ▼
[Embed Query]
    │
    ▼
[Query Pinecone]
    │
    └─► Returns top_k_dense Candidates
    │
    ▼
[Fetch Candidate Texts]
    │
    └─► From LocalChunkStore
    │
    ▼
[Rerank with LLM]
    │
    └─► Returns RerankResults (sorted by score)
    │
    ▼
[Assemble Context]
    │
    ├─► Fetch Primary Chunks
    ├─► Include Parent Chunks
    ├─► Expand Neighbors
    ├─► Order and Deduplicate
    ├─► Apply Token Budget
    └─► Generate Citations
    │
    ▼
ContextPack
```

## Neighbor Expansion

**Location**: `src/pipelines/chunking/retrieval_safety.py`

### Function: `expand_neighbors(chunk, all_chunks, config) -> List[Chunk]`

**Purpose**: Expands chunk with neighbors for retrieval safety

**Process**:
- Gets sibling chunks within same parent
- Gets neighbors from adjacent pages
- Returns list including original chunk + neighbors

## Citations

**Location**: `src/utils/io.py`

### Function: `format_citation(chunk) -> str`

**Format**: `"Doc: {doc_id} | Page: {page} | Section: {section}"`

**Includes**:
- Doc ID
- Page span
- Section label/header path
- Element-specific ranges (row_range, item_range) if available

## Performance Considerations

- **Batch Embedding**: Embeddings are batched for efficiency
- **Caching**: Reranking results are cached
- **Token Budgeting**: Context size is managed to stay within limits
- **Parallel Processing**: Multiple operations can run in parallel

## Next Steps

- **[Configuration](./08-configuration.md)** - Configure retrieval settings
- **[API Reference](./14-api-reference.md)** - Retrieval API details
- **[Troubleshooting](./15-troubleshooting.md)** - Common issues

