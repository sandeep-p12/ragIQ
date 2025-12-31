"""Test script for chunking pipeline.

This script tests the complete chunking process:
1. Check for cached markdown from LLM_FULL parsing (to avoid API costs)
2. If not found, parse document using LLM_FULL strategy and cache markdown
3. Run chunking on the markdown
4. Validate chunking results
"""

import logging
import sys
import tempfile
import json
import hashlib
import time
from pathlib import Path
from typing import Dict, List

# Add project root to path
_project_root = Path(__file__).parent.parent.parent.resolve()
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from src.config.chunking import ChunkConfig
from src.config.parsing import ParseForgeConfig
from src.config.parsing_strategies import StrategyEnum
from src.pipelines.chunking.chunking import process_document
from src.pipelines.parsing.parseforge import ParseForge
from src.schema.chunk import Chunk, ParentChunk
from src.schema.document import Document

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_chunking_pipeline():
    """Test the complete parsing and chunking pipeline."""
    
    print("=" * 80)
    print("RAG IQ - Chunking Pipeline Test")
    print("=" * 80)
    print()
    
    # Get the PDF file path
    pdf_path = _project_root / "src" / "data" / "parsing" / "Sample-Enhanced-Memo.pdf"
    
    if not pdf_path.exists():
        print(f"❌ ERROR: PDF file not found at {pdf_path}")
        print(f"   Please ensure the file exists at the expected location.")
        return False
    
    print(f"✓ Found PDF file: {pdf_path}")
    print(f"  File size: {pdf_path.stat().st_size / 1024:.2f} KB")
    print()
    
    # Test 1: Configuration Loading
    print("Test 1: Loading Configuration...")
    try:
        parse_config = ParseForgeConfig()
        chunk_config = ChunkConfig()
        
        print(f"  ✓ Parse Device: {parse_config.device}")
        print(f"  ✓ Parse Batch Size: {parse_config.batch_size}")
        print(f"  ✓ LLM Provider: {parse_config.llm_provider}")
        print(f"  ✓ LLM Model: {parse_config.llm_model}")
        print(f"  ✓ Chunk Prose Target: {chunk_config.prose_target_tokens} tokens")
        print(f"  ✓ Chunk Prose Overlap: {chunk_config.prose_overlap_tokens} tokens")
        print()
    except Exception as e:
        print(f"  ❌ ERROR: Failed to load configuration: {e}")
        logger.exception(e)
        return False
    
    # Test 2: ParseForge Initialization
    print("Test 2: Initializing ParseForge...")
    try:
        parser = ParseForge(config=parse_config)
        print("  ✓ ParseForge initialized successfully")
        print()
    except Exception as e:
        print(f"  ❌ ERROR: Failed to initialize ParseForge: {e}")
        logger.exception(e)
        return False
    
    # Setup cache directory
    cache_dir = Path(__file__).parent / "temp"
    cache_dir.mkdir(exist_ok=True)
    
    # Generate cache key for LLM_FULL strategy (same as parsing test)
    pdf_hash = hashlib.md5(str(pdf_path).encode()).hexdigest()[:8]
    config_hash = hashlib.md5(
        json.dumps({
            "llm_provider": parse_config.llm_provider,
            "llm_model": parse_config.llm_model,
            "generate_image_descriptions": parse_config.llm_provider != "none" and parse_config.llm_api_key is not None
        }, sort_keys=True).encode()
    ).hexdigest()[:8]
    cache_key = f"{pdf_path.stem}_llm_full_{pdf_hash}_{config_hash}"
    cache_file = cache_dir / f"{cache_key}.md"
    cache_meta_file = cache_dir / f"{cache_key}.meta.json"
    
    # Test 3: Get Markdown (from cache or generate)
    print("Test 3: Getting Markdown...")
    markdown = None
    document = None
    
    # Check if cached markdown exists
    if cache_file.exists() and cache_meta_file.exists():
        print(f"  ℹ Found cached markdown: {cache_file.name}")
        print("  ℹ Loading from cache to avoid API calls...")
        try:
            with open(cache_file, 'r', encoding='utf-8') as f:
                markdown = f.read()
            with open(cache_meta_file, 'r', encoding='utf-8') as f:
                cache_meta = json.load(f)
            print(f"  ✓ Loaded cached markdown (generated: {cache_meta.get('timestamp', 'unknown')})")
            print(f"  ✓ Markdown length: {len(markdown):,} characters")
            print(f"  ✓ Markdown lines: {len(markdown.splitlines()):,}")
            print()
            
            # Create a dummy document for stats (we don't need the full document for chunking)
            # But we'll still need to parse if we want document stats
            # For now, just use the markdown
        except Exception as e:
            print(f"  ⚠ Warning: Failed to load cached markdown: {e}")
            print("  ℹ Will parse document instead...")
            cache_file = None  # Force parsing
    
    # If no cached markdown, parse document
    if markdown is None:
        print("Test 3: Parsing Document with LLM_FULL strategy...")
        print(f"  Processing: {pdf_path.name}")
        print("  This may take a few moments...")
        print()
        
        try:
            document = parser.parse(
                str(pdf_path),
                strategy=StrategyEnum.LLM_FULL
            )
            
            # Validate document
            assert isinstance(document, Document), "Document is not a Document instance"
            assert hasattr(document, 'pages'), "Document missing 'pages' attribute"
            assert len(document.pages) > 0, "Document has no pages"
            
            print(f"  ✓ Document parsed successfully")
            print(f"  ✓ Document has {len(document.pages)} pages")
            print()
            
        except Exception as e:
            print(f"  ❌ ERROR: Failed to parse document: {e}")
            logger.exception(e)
            return False
        
        # Test 4: Markdown Generation
        print("Test 4: Converting to Markdown...")
        try:
            generate_image_descriptions = parse_config.llm_provider != "none" and parse_config.llm_api_key is not None
            
            print("  ℹ Generating markdown (this may make API calls)...")
            markdown = parser.to_markdown(
                document,
                generate_image_descriptions=generate_image_descriptions
            )
            
            assert markdown is not None, "Markdown is None"
            assert len(markdown) > 0, "Markdown is empty"
            
            print(f"  ✓ Markdown generated successfully")
            print(f"  ✓ Markdown length: {len(markdown):,} characters")
            print(f"  ✓ Markdown lines: {len(markdown.splitlines()):,}")
            print()
            
            # Save markdown to cache
            print(f"  ℹ Saving markdown to cache: {cache_file.name}")
            with open(cache_file, 'w', encoding='utf-8') as f:
                f.write(markdown)
            with open(cache_meta_file, 'w', encoding='utf-8') as f:
                json.dump({
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "strategy": "LLM_FULL",
                    "pdf_path": str(pdf_path),
                    "pdf_hash": pdf_hash,
                    "config_hash": config_hash,
                    "llm_provider": parse_config.llm_provider,
                    "llm_model": parse_config.llm_model,
                    "generate_image_descriptions": generate_image_descriptions,
                    "pages": len(document.pages) if document else 0
                }, f, indent=2)
            print(f"  ✓ Markdown cached: {cache_file.name}")
            print()
            
        except Exception as e:
            print(f"  ❌ ERROR: Failed to convert to markdown: {e}")
            logger.exception(e)
            return False
    
    # Save markdown to temp file for chunking
    with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False, encoding='utf-8') as tmp_file:
        tmp_file.write(markdown)
        markdown_path = tmp_file.name
    
    print(f"  ✓ Markdown ready for chunking: {markdown_path}")
    print()
    
    # Test 5: Chunking
    print("Test 5: Chunking Document...")
    print("  This may take a few moments...")
    print()
    
    try:
        doc_id = "test_doc_llm_full"
        
        children, parents, stats = process_document(
            markdown_path,
            chunk_config,
            doc_id
        )
        
        # Validate results
        assert isinstance(children, list), "Children is not a list"
        assert isinstance(parents, list), "Parents is not a list"
        assert isinstance(stats, dict), "Stats is not a dict"
        assert len(children) > 0, "No children chunks generated"
        
        print(f"  ✓ Chunking completed successfully")
        print(f"  ✓ Children chunks: {len(children)}")
        print(f"  ✓ Parent chunks: {len(parents)}")
        print()
        
        # Validate chunk structure
        print("Test 6: Validating Chunk Structure...")
        for i, chunk in enumerate(children[:5]):  # Check first 5 chunks
            assert isinstance(chunk, Chunk), f"Chunk {i} is not a Chunk instance"
            assert hasattr(chunk, 'chunk_id'), f"Chunk {i} missing 'chunk_id'"
            assert hasattr(chunk, 'doc_id'), f"Chunk {i} missing 'doc_id'"
            assert hasattr(chunk, 'text_for_embedding'), f"Chunk {i} missing 'text_for_embedding'"
            assert hasattr(chunk, 'raw_md_fragment'), f"Chunk {i} missing 'raw_md_fragment'"
            assert chunk.doc_id == doc_id, f"Chunk {i} has incorrect doc_id"
            assert chunk.text_for_embedding is not None and len(chunk.text_for_embedding) > 0, \
                f"Chunk {i} has empty text_for_embedding"
        
        print("  ✓ Chunk structure validation passed")
        print()
        
        # Validate parent structure
        if len(parents) > 0:
            print("Test 7: Validating Parent Chunk Structure...")
            for i, parent in enumerate(parents[:3]):  # Check first 3 parents
                assert isinstance(parent, ParentChunk), f"Parent {i} is not a ParentChunk instance"
                assert hasattr(parent, 'chunk_id'), f"Parent {i} missing 'chunk_id'"
                assert hasattr(parent, 'doc_id'), f"Parent {i} missing 'doc_id'"
                # Check for child_ids attribute
                assert hasattr(parent, 'child_ids'), f"Parent {i} missing 'child_ids'"
                assert len(parent.child_ids) > 0, f"Parent {i} has no children"
            
            print("  ✓ Parent chunk structure validation passed")
            print()
        
        # Display statistics
        print("Test 8: Chunking Statistics...")
        print(f"  ✓ Total children chunks: {len(children)}")
        print(f"  ✓ Total parent chunks: {len(parents)}")
        
        if stats:
            print(f"  ✓ Average tokens per chunk: {stats.get('avg_tokens_per_chunk', 0):.0f}")
            print(f"  ✓ Total elements processed: {stats.get('total_elements', 0)}")
            print(f"  ✓ Total page blocks: {stats.get('total_page_blocks', 0)}")
        
        # Token distribution
        token_counts = [chunk.token_count for chunk in children if hasattr(chunk, 'token_count') and chunk.token_count]
        if token_counts:
            print(f"  ✓ Min tokens: {min(token_counts)}")
            print(f"  ✓ Max tokens: {max(token_counts)}")
            print(f"  ✓ Avg tokens: {sum(token_counts) / len(token_counts):.0f}")
        
        # Element type distribution
        element_types = {}
        for chunk in children:
            elem_type = chunk.element_type if hasattr(chunk, 'element_type') else 'unknown'
            element_types[elem_type] = element_types.get(elem_type, 0) + 1
        
        print(f"  ✓ Element type distribution:")
        for elem_type, count in sorted(element_types.items()):
            print(f"    - {elem_type}: {count}")
        
        print()
        
        # Show sample chunks
        print("Test 9: Sample Chunks Preview...")
        print("  First 3 children chunks:")
        for i, chunk in enumerate(children[:3], 1):
            print(f"\n  Chunk {i}:")
            print(f"    - ID: {chunk.chunk_id[:50]}...")
            print(f"    - Type: {chunk.element_type if hasattr(chunk, 'element_type') else 'unknown'}")
            print(f"    - Tokens: {chunk.token_count if hasattr(chunk, 'token_count') else 'N/A'}")
            print(f"    - Pages: {chunk.page_nos if hasattr(chunk, 'page_nos') else 'N/A'}")
            text_preview = chunk.text_for_embedding[:100] if chunk.text_for_embedding else ""
            print(f"    - Preview: {text_preview}...")
        
        if len(parents) > 0:
            print(f"\n  First parent chunk:")
            parent = parents[0]
            print(f"    - ID: {parent.chunk_id[:50]}...")
            print(f"    - Children: {len(parent.child_ids)}")
            print(f"    - Pages: {parent.page_nos if hasattr(parent, 'page_nos') else 'N/A'}")
        
        print()
        
    except Exception as e:
        print(f"  ❌ ERROR: Failed to chunk document: {e}")
        logger.exception(e)
        return False
    
    # Summary
    print("=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    print("✓ All tests passed successfully!")
    print()
    print("Pipeline Statistics:")
    if document:
        print(f"  - Parsed Pages: {len(document.pages)}")
    else:
        print(f"  - Parsed Pages: (used cached markdown)")
    print(f"  - Markdown Length: {len(markdown):,} characters")
    print(f"  - Markdown Lines: {len(markdown.splitlines()):,}")
    print(f"  - Children Chunks: {len(children)}")
    print(f"  - Parent Chunks: {len(parents)}")
    if 'token_counts' in locals() and token_counts:
        print(f"  - Avg Tokens/Chunk: {sum(token_counts) / len(token_counts):.0f}")
    print()
    print("Component Status:")
    print(f"  - ParseForge: ✓ Initialized")
    print(f"  - LLM Provider: ✓ {'Available' if parse_config.llm_provider != 'none' and parse_config.llm_api_key else 'Not configured'}")
    print(f"  - Chunking Pipeline: ✓ Working")
    print(f"  - Markdown Source: {'Cached' if cache_file.exists() else 'Generated'}")
    print()
    print("=" * 80)
    print("✅ Parsing and chunking pipeline test completed successfully!")
    print("=" * 80)
    
    return True


if __name__ == "__main__":
    success = test_chunking_pipeline()
    sys.exit(0 if success else 1)

