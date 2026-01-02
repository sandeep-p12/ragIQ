# Troubleshooting Guide

## Common Issues

### Installation Issues

#### Import Errors

**Problem**: `ModuleNotFoundError` or import errors

**Solutions**:
1. Ensure you're in the project root directory
2. Install dependencies: `uv sync` or `pip install -e .`
3. Check Python version: `python --version` (should be >= 3.10)
4. Verify virtual environment is activated

#### Missing Dependencies

**Problem**: Missing optional dependencies (OCR, layout detection)

**Solutions**:
1. Install optional dependencies:
   ```bash
   pip install onnxruntime doctr
   ```
2. Download required model files to `src/ai_models/`
3. System will work without models (with reduced features)

### Configuration Issues

#### API Key Errors

**Problem**: `OPENAI_API_KEY not found` or similar errors

**Solutions**:
1. Check `.env` file exists in project root
2. Verify API key is set: `OPENAI_API_KEY=sk-...`
3. Ensure no extra spaces or quotes around the key
4. Restart the application after changing `.env`

#### Azure OpenAI Authentication Failed

**Problem**: Azure AD authentication fails

**Solutions**:
1. Ensure Azure CLI is logged in: `az login`
2. Check Managed Identity configuration (for Azure-hosted deployments)
3. Verify service principal credentials
4. Try API key fallback: `AZURE_OPENAI_USE_AZURE_AD=false`

#### Configuration Not Loading

**Problem**: Configuration values not taking effect

**Solutions**:
1. Check environment variable names match exactly (case-sensitive)
2. Verify `.env` file is in project root
3. Restart application after changing configuration
4. Use `from_env()` methods for retrieval config

### Parsing Issues

#### PDF Parsing Fails

**Problem**: PDF parsing errors or empty results

**Solutions**:
1. Try different parsing strategy:
   - `FAST` for text-based PDFs
   - `HI_RES` for scanned documents
   - `AUTO` for mixed quality
2. Check PDF is not corrupted
3. Verify PDF has extractable text (for FAST strategy)
4. Enable OCR models for scanned documents

#### OCR Not Working

**Problem**: OCR detection fails or returns empty results

**Solutions**:
1. Verify Doctr models are in `src/ai_models/`:
   - `fast_base.pt`
   - `crnn_vgg16_bn.pt`
2. Check device configuration: `PARSEFORGE_DEVICE=cpu` or `cuda`
3. Try lower batch size: `PARSEFORGE_BATCH_SIZE=10`
4. Check image quality (low quality images may fail)

#### Layout Detection Fails

**Problem**: Layout detection returns no results

**Solutions**:
1. Verify YOLO model is in `src/ai_models/`:
   - `doclayout_yolo_ft.pt`
2. Lower confidence threshold in code
3. Check device configuration
4. Verify image dimensions are reasonable

### Chunking Issues

#### Chunks Too Large

**Problem**: Chunks exceed token limits

**Solutions**:
1. Lower `max_chunk_tokens_hard` in ChunkConfig
2. Reduce `prose_target_tokens`
3. Enable more aggressive splitting
4. Check token counting is accurate

#### Chunks Too Small

**Problem**: Many tiny chunks

**Solutions**:
1. Increase `min_chunk_tokens` in ChunkConfig
2. Enable cross-page merging
3. Adjust merge aggressiveness
4. Check if markdown repair is too aggressive

#### Missing Parent Chunks

**Problem**: No parent chunks created

**Solutions**:
1. Check document has headings (for heading-based grouping)
2. Lower `structure_confidence_threshold` for soft-section grouping
3. Verify `parent_heading_level` matches document structure
4. Check chunking statistics for grouping information

### Indexing Issues

#### Embedding Errors

**Problem**: Embedding API calls fail

**Solutions**:
1. Check API key is valid
2. Verify rate limits not exceeded
3. Reduce `batch_size` in EmbeddingConfig
4. Check network connectivity
5. For Azure: Verify endpoint and deployment name

#### Pinecone Errors

**Problem**: Pinecone upsert or query fails

**Solutions**:
1. Verify `PINECONE_API_KEY` is set
2. Check index name exists: `PINECONE_INDEX_NAME`
3. Verify index dimension matches embedding dimension
4. Check namespace is correct: `PINECONE_NAMESPACE`
5. Verify Pinecone service is accessible

#### Index Creation Fails

**Problem**: Cannot create Pinecone index

**Solutions**:
1. Check Pinecone account has index creation permissions
2. Verify index name is unique
3. Check dimension matches embedding model
4. Verify Pinecone plan supports index creation

### Retrieval Issues

#### No Results Returned

**Problem**: Retrieval returns empty results

**Solutions**:
1. Verify chunks are indexed (check Pinecone index)
2. Check filters are not too restrictive
3. Verify query embedding is generated correctly
4. Check `top_k_dense` is large enough
5. Verify namespace matches indexing namespace

#### Reranking Fails

**Problem**: LLM reranking errors

**Solutions**:
1. Check LLM API key is valid
2. Verify model name/deployment name is correct
3. Check rate limits
4. Reduce `max_candidates_to_rerank`
5. Verify JSON response format

#### Context Too Large

**Problem**: Context pack exceeds token budget

**Solutions**:
1. Reduce `final_max_tokens` in RetrievalConfig
2. Reduce `neighbor_same_page` and `neighbor_cross_page`
3. Disable `include_parents`
4. Reduce `return_top_n` in RerankConfig

### Performance Issues

#### Slow Parsing

**Problem**: Document parsing is very slow

**Solutions**:
1. Use `FAST` strategy for text-based PDFs
2. Reduce `batch_size` if memory constrained
3. Use GPU if available: `PARSEFORGE_DEVICE=cuda`
4. Disable image descriptions if not needed
5. Use page range limits for testing

#### Slow Embedding

**Problem**: Embedding generation is slow

**Solutions**:
1. Increase `batch_size` in EmbeddingConfig
2. Use faster embedding model
3. Check network latency
4. Verify caching is working
5. Use Azure OpenAI if closer to your location

#### Memory Issues

**Problem**: Out of memory errors

**Solutions**:
1. Reduce `batch_size` in all configs
2. Process documents in smaller chunks
3. Use CPU instead of GPU if GPU memory limited
4. Close other applications
5. Process fewer pages at a time

## Debugging Tips

### Enable Logging

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

### Check Configuration

```python
from src.config.parsing import ParseForgeConfig
config = ParseForgeConfig()
print(config.model_dump())
```

### Verify Models

```python
from pathlib import Path
model_dir = Path("src/ai_models")
print(f"Models: {list(model_dir.glob('*.pt'))}")
```

### Test Components

```python
# Test embedding
from src.providers.embedding.openai_embedding import OpenAIEmbeddingProvider
provider = OpenAIEmbeddingProvider()
result = provider.embed_query("test")
print(f"Embedding dimension: {len(result)}")

# Test LLM
from src.providers.llm.openai_llm import OpenAILLMProvider
provider = OpenAILLMProvider()
result = provider.generate("Hello")
print(f"Response: {result}")
```

## Getting Help

1. Check documentation in `documentation/` folder
2. Review error messages carefully
3. Check logs for detailed error information
4. Verify configuration matches examples
5. Test with simple examples first

## Next Steps

- **[Configuration](./08-configuration.md)** - Review configuration options
- **[API Reference](./14-api-reference.md)** - Check API usage
- **[Setup](./02-setup.md)** - Revisit setup if needed

