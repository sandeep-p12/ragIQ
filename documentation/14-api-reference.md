# API Reference

## ParseForge

**Location**: `src/pipelines/parsing/parseforge.py`

### Class: `ParseForge`

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

**Example**:
```python
from src.config.parsing import ParseForgeConfig
from src.config.parsing_strategies import StrategyEnum
from src.pipelines.parsing.parseforge import ParseForge

config = ParseForgeConfig(device="cpu", llm_provider="openai")
parser = ParseForge(config)

document = parser.parse("document.pdf", strategy=StrategyEnum.AUTO)
markdown = parser.to_markdown(document, generate_image_descriptions=True)
```

## Chunking

**Location**: `src/pipelines/chunking/chunking.py`

### Function: `process_document`

```python
def process_document(
    file_path: str,
    config: ChunkConfig,
    doc_id: str
) -> Tuple[List[Chunk], List[ParentChunk], Dict[str, Any]]
```

**Example**:
```python
from src.config.chunking import ChunkConfig
from src.pipelines.chunking.chunking import process_document

config = ChunkConfig(
    prose_target_tokens=512,
    parent_heading_level=2
)

children, parents, stats = process_document("document.md", config, "doc_1")
```

## Retrieval

**Location**: `src/pipelines/retrieval/retrieval.py`

### Function: `ingest_from_chunking_outputs`

```python
def ingest_from_chunking_outputs(
    children_jsonl_path: str,
    parents_jsonl_path: str,
    doc_id: str,
    cfg: RetrievalConfig = None
) -> Dict[str, Any]
```

**Example**:
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

### Function: `retrieve`

```python
def retrieve(
    query: str,
    filters: Optional[Dict[str, Any]] = None,
    cfg: RetrievalConfig = None,
    all_chunks: Optional[List[Dict[str, Any]]] = None,
    llm_config: Optional[ParseForgeConfig] = None
) -> ContextPack
```

**Example**:
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
print(result.context_pack)
```

## RAGOrchestrator

**Location**: `src/pipelines/orchestrator.py`

### Class: `RAGOrchestrator`

```python
class RAGOrchestrator:
    def __init__(
        self,
        parsing_config: Optional[ParseForgeConfig] = None,
        chunking_config: Optional[ChunkConfig] = None,
        retrieval_config: Optional[RetrievalConfig] = None,
        progress_callback: Optional[callable] = None
    )
    
    def parse(
        self,
        file_path: str,
        strategy: StrategyEnum = StrategyEnum.AUTO,
        start_page: Optional[int] = None,
        end_page: Optional[int] = None
    ) -> Document
    
    def parse_to_markdown(
        self,
        file_path: str,
        strategy: StrategyEnum = StrategyEnum.AUTO,
        generate_image_descriptions: bool = True,
        start_page: Optional[int] = None,
        end_page: Optional[int] = None
    ) -> str
    
    def chunk(
        self,
        markdown_path: str,
        doc_id: str
    ) -> Tuple[List[Chunk], List[ParentChunk], Dict[str, Any]]
    
    def index(
        self,
        children: List[Chunk],
        parents: List[ParentChunk],
        doc_id: str,
        output_dir: Optional[str] = None
    ) -> Dict[str, Any]
    
    def query(
        self,
        query: str,
        filters: Optional[Dict[str, Any]] = None,
        all_chunks: Optional[List[Dict[str, Any]]] = None
    ) -> ContextPack
    
    def pipeline(
        self,
        file_path: str,
        doc_id: str,
        strategy: StrategyEnum = StrategyEnum.AUTO,
        generate_image_descriptions: bool = True,
        index_chunks: bool = True
    ) -> Dict[str, Any]
```

**Example**:
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

## Providers

### OpenAILLMProvider

**Location**: `src/providers/llm/openai_llm.py`

```python
class OpenAILLMProvider:
    def __init__(self, config: Optional[ParseForgeConfig] = None)
    
    def generate(
        self,
        prompt: str,
        model: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: Optional[int] = None
    ) -> str
    
    def generate_vision(
        self,
        prompt: str,
        images: List[Any],
        model: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: Optional[int] = None
    ) -> str
```

### OpenAIEmbeddingProvider

**Location**: `src/providers/embedding/openai_embedding.py`

```python
class OpenAIEmbeddingProvider:
    def __init__(self, config: EmbeddingConfig = None)
    
    def embed_texts(self, texts: List[str]) -> List[List[float]]
    
    def embed_query(self, query: str) -> List[float]
```

## Storage

### PineconeVectorStore

**Location**: `src/storage/vector/pinecone.py`

```python
class PineconeVectorStore:
    def __init__(self, config: PineconeConfig)
    
    def upsert(
        self,
        vectors: List[VectorRecord],
        namespace: str
    ) -> None
    
    def query(
        self,
        vector: List[float],
        top_k: int,
        filters: Optional[Dict[str, Any]] = None,
        namespace: str = "children"
    ) -> List[Candidate]
```

### LocalChunkStore

**Location**: `src/storage/chunk/local.py`

```python
class LocalChunkStore:
    def put_chunks(
        self,
        children: List[Dict[str, Any]],
        parents: List[Dict[str, Any]]
    ) -> None
    
    def get_chunk(
        self,
        doc_id: str,
        chunk_id: str,
        is_parent: bool = False
    ) -> Optional[Dict[str, Any]]
    
    def get_chunks_bulk(
        self,
        keys: List[Tuple[str, str, bool]]
    ) -> List[Optional[Dict[str, Any]]]
```

## Configuration

### ParseForgeConfig

**Location**: `src/config/parsing.py`

```python
class ParseForgeConfig(BaseSettings):
    device: Literal["cpu", "cuda", "mps", "coreml"] = "cpu"
    batch_size: int = 50
    llm_provider: Literal["openai", "azure_openai", "none"] = "openai"
    llm_model: str = "gpt-4o"
    # ... more fields
```

### ChunkConfig

**Location**: `src/config/chunking.py`

```python
@dataclass
class ChunkConfig:
    prose_target_tokens: int = 512
    prose_overlap_tokens: int = 50
    parent_heading_level: int = 2
    # ... more fields
```

### RetrievalConfig

**Location**: `src/config/retrieval.py`

```python
@dataclass
class RetrievalConfig:
    neighbor_same_page: int = 1
    neighbor_cross_page: int = 2
    include_parents: bool = True
    final_max_tokens: int = 12000
    # ... more fields
    
    @classmethod
    def from_env(cls, **overrides) -> RetrievalConfig
```

## Complete Example

```python
from src.pipelines.orchestrator import RAGOrchestrator
from src.config.parsing_strategies import StrategyEnum

# Initialize orchestrator
orchestrator = RAGOrchestrator()

# Run complete pipeline
results = orchestrator.pipeline(
    file_path="document.pdf",
    doc_id="doc_1",
    strategy=StrategyEnum.AUTO,
    generate_image_descriptions=True,
    index_chunks=True
)

# Query
result = orchestrator.query(
    query="What is the main topic?",
    filters={"doc_id": "doc_1"}
)

print(f"Retrieved {len(result.selected_chunks)} chunks")
print(result.context_pack)
```

## Next Steps

- **[Usage Examples](./02-setup.md)** - More usage examples
- **[Troubleshooting](./15-troubleshooting.md)** - Common API issues

