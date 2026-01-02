# Providers

## Overview

Providers are external service integrations that handle LLM, embeddings, OCR, and layout detection.

## LLM Provider

**Location**: `src/providers/llm/openai_llm.py`

### Class: `OpenAILLMProvider`

**Supports**: OpenAI and Azure OpenAI

**Initialization**:
```python
from src.config.parsing import ParseForgeConfig
from src.providers.llm.openai_llm import OpenAILLMProvider

config = ParseForgeConfig(llm_provider="openai")
provider = OpenAILLMProvider(config)
```

**Key Methods**:
- `generate(prompt, model, temperature, max_tokens) -> str`: Text generation
- `generate_vision(prompt, images, model, temperature, max_tokens) -> str`: Vision generation

**Features**:
- Supports both OpenAI and Azure OpenAI
- Azure AD authentication using DefaultAzureCredential (default)
- API key authentication fallback
- Vision support for image processing
- Rate limit and API error handling

**Vision Support**:
- Converts images to base64
- Supports PIL Images, base64 strings, bytes
- Used for image descriptions and table extraction

## Embedding Provider

**Location**: `src/providers/embedding/openai_embedding.py`

### Class: `OpenAIEmbeddingProvider`

**Supports**: OpenAI and Azure OpenAI

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

**Caching**:
- Caches embeddings by model name + text
- Reduces API calls for repeated texts
- In-memory cache (cleared on restart)

## OCR Provider

**Location**: `src/providers/ocr/doctr.py`

### Class: `DoctrOCR`

**Purpose**: Text detection using Doctr OCR

**Initialization**:
```python
from src.providers.ocr.doctr import DoctrOCR

ocr = DoctrOCR()
```

**Key Methods**:
- `detect_text(images) -> List[TextDetection]`: Detects text in images

**TextDetection Dataclass**:
- `text`: Detected text
- `bbox`: Bounding box
- `confidence`: Detection confidence
- `page_index`: Page index
- `dimensions`: Image dimensions

**Models Required**:
- `fast_base.pt`: Doctr detection model
- `crnn_vgg16_bn.pt`: Doctr recognition model

## Layout Provider

**Location**: `src/providers/layout/yolo.py`

### Class: `YOLOLayoutDetector`

**Purpose**: Layout detection using YOLO

**Initialization**:
```python
from src.providers.layout.yolo import YOLOLayoutDetector

detector = YOLOLayoutDetector(device="cpu", confidence_threshold=0.2)
```

**Key Methods**:
- `detect_layout(image, page_index) -> List[LayoutDetectionOutput]`: Detects layout elements

**Features**:
- Supports devices: cpu, cuda, mps, coreml
- Configurable confidence threshold (default: 0.2)
- Image size: 1024 for prediction

**Label Mapping** (doclayout_yolo_ft.pt model):
- 0: "title" → `BlockType.TITLE`
- 1: "plain text" → `BlockType.TEXT`
- 2: "abandon" → skipped
- 3: "figure" → `BlockType.IMAGE`
- 4: "figure_caption" → `BlockType.CAPTION`
- 5: "table" → `BlockType.TABLE`
- 6: "table_caption" → `BlockType.CAPTION`
- 7: "table_footnote" → `BlockType.FOOTNOTE`
- 8: "isolate_formula" → `BlockType.TEXT`
- 9: "formula_caption" → `BlockType.TEXT`

**LayoutDetectionOutput Dataclass**:
- `bbox`: Normalized bounding box (BBox)
- `category_id`: YOLO category ID
- `score`: Confidence score
- `block_type`: Mapped BlockType

**Model Required**:
- `doclayout_yolo_ft.pt`: YOLO layout detection model

## Provider Interfaces

**Location**: `src/core/interfaces.py`

### Protocols Defined:
- `EmbeddingProvider`: `embed_texts()`, `embed_query()`
- `VectorStore`: `upsert()`, `query()`
- `ChunkStore`: `put_chunks()`, `get_chunk()`, `get_chunks_bulk()`
- `Reranker`: `rerank()`
- `ContextAssembler`: `assemble()`
- `LLMProvider`: `generate()`
- `OCRProvider`: `detect_text()`
- `LayoutProvider`: `detect_layout()`

**Design Pattern**: Uses Python Protocols for duck typing, enabling easy swapping of implementations.

## Error Handling

All providers handle errors gracefully:
- Missing API keys: Logs warning, disables operations
- Rate limits: Exponential backoff retry
- API errors: Proper error messages
- Missing models: Disables features, continues with available features

## Next Steps

- **[Storage](./11-storage.md)** - Storage backends
- **[Configuration](./08-configuration.md)** - Configure providers
- **[Troubleshooting](./15-troubleshooting.md)** - Provider issues

