"""Test script for full parsing pipeline.

This script tests all parsing strategies on a sample PDF document
to ensure all components are properly connected and working without errors.
Each strategy's markdown is cached separately for reuse.
"""

import logging
import sys
import hashlib
import json
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

# Add project root to path
_project_root = Path(__file__).parent.parent.parent.resolve()
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from src.config.parsing import ParseForgeConfig
from src.config.parsing_strategies import StrategyEnum
from src.pipelines.parsing.parseforge import ParseForge
from src.schema.document import Document

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# All strategies to test
ALL_STRATEGIES = [
    StrategyEnum.FAST,
    StrategyEnum.HI_RES,
    StrategyEnum.AUTO,
    StrategyEnum.LLM_FULL,
]


def get_cache_key(pdf_path: Path, config: ParseForgeConfig, strategy: StrategyEnum) -> str:
    """Generate cache key for a specific strategy."""
    pdf_hash = hashlib.md5(str(pdf_path).encode()).hexdigest()[:8]
    config_hash = hashlib.md5(
        json.dumps({
            "llm_provider": config.llm_provider,
            "llm_model": config.llm_model,
            "generate_image_descriptions": config.llm_provider != "none" and config.llm_api_key is not None
        }, sort_keys=True).encode()
    ).hexdigest()[:8]
    strategy_name = strategy.value
    return f"{pdf_path.stem}_{strategy_name}_{pdf_hash}_{config_hash}"


def test_strategy(
    parser: ParseForge,
    pdf_path: Path,
    config: ParseForgeConfig,
    strategy: StrategyEnum,
    cache_dir: Path
) -> Tuple[bool, Optional[Document], Optional[str], Dict]:
    """
    Test a single parsing strategy.
    
    Returns:
        Tuple of (success, document, markdown, stats)
    """
    strategy_name = strategy.value.upper()
    print(f"\n{'=' * 80}")
    print(f"Testing Strategy: {strategy_name}")
    print(f"{'=' * 80}")
    print()
    
    stats = {
        "strategy": strategy_name,
        "success": False,
        "pages": 0,
        "blocks": 0,
        "markdown_length": 0,
        "markdown_lines": 0,
        "cached": False,
        "error": None
    }
    
    # Setup cache files
    cache_key = get_cache_key(pdf_path, config, strategy)
    cache_file = cache_dir / f"{cache_key}.md"
    cache_meta_file = cache_dir / f"{cache_key}.meta.json"
    
    try:
        # Step 1: Parse Document
        print(f"Step 1: Parsing Document with {strategy_name} strategy...")
        print(f"  Processing: {pdf_path.name}")
        print("  This may take a few moments...")
        print()
        
        document = parser.parse(
            str(pdf_path),
            strategy=strategy
        )
        
        # Validate document
        assert isinstance(document, Document), "Document is not a Document instance"
        assert hasattr(document, 'pages'), "Document missing 'pages' attribute"
        assert len(document.pages) > 0, "Document has no pages"
        
        stats["pages"] = len(document.pages)
        total_blocks = sum(len(page.blocks) for page in document.pages)
        stats["blocks"] = total_blocks
        
        print(f"  ✓ Document parsed successfully")
        print(f"  ✓ Document has {stats['pages']} pages")
        print(f"  ✓ Document has {stats['blocks']} total blocks")
        
        # Count block types
        block_types = {}
        for page in document.pages:
            for block in page.blocks:
                block_type = block.block_type.value if hasattr(block.block_type, 'value') else str(block.block_type)
                block_types[block_type] = block_types.get(block_type, 0) + 1
        
        print("  ✓ Block type distribution:")
        for block_type, count in sorted(block_types.items()):
            print(f"    - {block_type}: {count}")
        print()
        
        # Step 2: Generate Markdown
        print(f"Step 2: Converting to Markdown...")
        
        # Check if cached markdown exists
        if cache_file.exists() and cache_meta_file.exists():
            print(f"  ℹ Found cached markdown: {cache_file.name}")
            print("  ℹ Loading from cache to avoid API calls...")
            with open(cache_file, 'r', encoding='utf-8') as f:
                markdown = f.read()
            with open(cache_meta_file, 'r', encoding='utf-8') as f:
                cache_meta = json.load(f)
            print(f"  ✓ Loaded cached markdown (generated: {cache_meta.get('timestamp', 'unknown')})")
            stats["cached"] = True
        else:
            # Check if LLM is configured for image descriptions
            generate_image_descriptions = config.llm_provider != "none" and config.llm_api_key is not None
            
            print("  ℹ Generating markdown (this may make API calls)...")
            markdown = parser.to_markdown(
                document,
                generate_image_descriptions=generate_image_descriptions
            )
            
            # Validate markdown
            assert markdown is not None, "Markdown is None"
            assert len(markdown) > 0, "Markdown is empty"
            
            # Save to cache immediately after successful generation
            print(f"  ℹ Saving markdown to cache: {cache_file.name}")
            with open(cache_file, 'w', encoding='utf-8') as f:
                f.write(markdown)
            with open(cache_meta_file, 'w', encoding='utf-8') as f:
                json.dump({
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "strategy": strategy_name,
                    "pdf_path": str(pdf_path),
                    "pdf_hash": hashlib.md5(str(pdf_path).encode()).hexdigest()[:8],
                    "config_hash": hashlib.md5(
                        json.dumps({
                            "llm_provider": config.llm_provider,
                            "llm_model": config.llm_model,
                            "generate_image_descriptions": generate_image_descriptions
                        }, sort_keys=True).encode()
                    ).hexdigest()[:8],
                    "llm_provider": config.llm_provider,
                    "llm_model": config.llm_model,
                    "generate_image_descriptions": generate_image_descriptions,
                    "pages": stats["pages"],
                    "blocks": stats["blocks"]
                }, f, indent=2)
            print(f"  ✓ Markdown generated and cached: {cache_file.name}")
            stats["cached"] = False
        
        stats["markdown_length"] = len(markdown)
        stats["markdown_lines"] = len(markdown.splitlines())
        
        print(f"  ✓ Markdown length: {stats['markdown_length']:,} characters")
        print(f"  ✓ Markdown lines: {stats['markdown_lines']:,}")
        print()
        
        # Show preview
        preview_lines = markdown.splitlines()[:15]
        print("  Preview (first 15 lines):")
        print("  " + "-" * 76)
        for i, line in enumerate(preview_lines, 1):
            print(f"  {i:2d} | {line[:74]}")
        print("  " + "-" * 76)
        print()
        
        stats["success"] = True
        print(f"✓ Strategy {strategy_name} completed successfully!")
        print()
        
        return True, document, markdown, stats
        
    except AssertionError as e:
        error_msg = f"Validation failed: {e}"
        print(f"  ❌ ERROR: {error_msg}")
        logger.exception(e)
        stats["error"] = error_msg
        return False, None, None, stats
    except Exception as e:
        error_msg = f"Failed: {e}"
        print(f"  ❌ ERROR: {error_msg}")
        logger.exception(e)
        stats["error"] = error_msg
        return False, None, None, stats


def test_parsing_pipeline():
    """Test all parsing strategies on the sample PDF."""
    
    print("=" * 80)
    print("RAG IQ - Full Parsing Pipeline Test (All Strategies)")
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
    
    # Setup cache directory
    cache_dir = Path(__file__).parent / "temp"
    cache_dir.mkdir(exist_ok=True)
    print(f"✓ Cache directory: {cache_dir}")
    print()
    
    # Test 1: Configuration Loading
    print("Test 1: Loading Configuration...")
    try:
        config = ParseForgeConfig()
        print(f"  ✓ Device: {config.device}")
        print(f"  ✓ Batch Size: {config.batch_size}")
        print(f"  ✓ LLM Provider: {config.llm_provider}")
        print(f"  ✓ LLM Model: {config.llm_model}")
        print(f"  ✓ Model Directory: {config.model_dir}")
        print()
    except Exception as e:
        print(f"  ❌ ERROR: Failed to load configuration: {e}")
        logger.exception(e)
        return False
    
    # Test 2: ParseForge Initialization
    print("Test 2: Initializing ParseForge...")
    try:
        parser = ParseForge(config=config)
        print("  ✓ ParseForge initialized successfully")
        print()
    except Exception as e:
        print(f"  ❌ ERROR: Failed to initialize ParseForge: {e}")
        logger.exception(e)
        return False
    
    # Test 3: Test All Strategies
    print("Test 3: Testing All Parsing Strategies...")
    print(f"  Strategies to test: {', '.join(s.value for s in ALL_STRATEGIES)}")
    print()
    
    results = {}
    all_success = True
    
    for strategy in ALL_STRATEGIES:
        success, document, markdown, stats = test_strategy(
            parser, pdf_path, config, strategy, cache_dir
        )
        results[strategy.value] = stats
        
        if not success:
            all_success = False
            print(f"⚠ Strategy {strategy.value.upper()} failed, continuing with other strategies...")
            print()
    
    # Test 4: Component Integration Check
    print("=" * 80)
    print("Test 4: Checking Component Integration...")
    print("=" * 80)
    try:
        # Check if formatters are initialized
        assert hasattr(parser, 'image_formatter'), "Image formatter not initialized"
        assert hasattr(parser, 'table_formatter'), "Table formatter not initialized"
        
        print("  ✓ Image formatter initialized")
        print("  ✓ Table formatter initialized")
        
        # Check if providers are accessible
        if parser.image_formatter:
            assert hasattr(parser.image_formatter, 'llm_provider'), "Image formatter missing LLM provider"
            print("  ✓ Image formatter has LLM provider")
        
        if parser.table_formatter:
            assert hasattr(parser.table_formatter, 'llm_provider'), "Table formatter missing LLM provider"
            print("  ✓ Table formatter has LLM provider")
        
        print()
    except AssertionError as e:
        print(f"  ❌ ERROR: Integration check failed: {e}")
        logger.exception(e)
        all_success = False
    except Exception as e:
        print(f"  ❌ ERROR: Unexpected error during integration check: {e}")
        logger.exception(e)
        all_success = False
    
    # Summary
    print("=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    
    successful_strategies = [s for s, stats in results.items() if stats.get("success", False)]
    failed_strategies = [s for s, stats in results.items() if not stats.get("success", False)]
    
    print(f"\nStrategies Tested: {len(results)}")
    print(f"  ✓ Successful: {len(successful_strategies)}")
    print(f"  ❌ Failed: {len(failed_strategies)}")
    print()
    
    if successful_strategies:
        print("Successful Strategies:")
        for strategy_name in successful_strategies:
            stats = results[strategy_name]
            print(f"  ✓ {strategy_name.upper()}:")
            print(f"    - Pages: {stats.get('pages', 0)}")
            print(f"    - Blocks: {stats.get('blocks', 0)}")
            print(f"    - Markdown: {stats.get('markdown_length', 0):,} chars, {stats.get('markdown_lines', 0):,} lines")
            print(f"    - Cached: {'Yes' if stats.get('cached', False) else 'No'}")
        print()
    
    if failed_strategies:
        print("Failed Strategies:")
        for strategy_name in failed_strategies:
            stats = results[strategy_name]
            print(f"  ❌ {strategy_name.upper()}: {stats.get('error', 'Unknown error')}")
        print()
    
    print("Component Status:")
    print(f"  - ParseForge: ✓ Initialized")
    print(f"  - Image Formatter: ✓ {'Initialized' if parser.image_formatter else 'Not initialized'}")
    print(f"  - Table Formatter: ✓ {'Initialized' if parser.table_formatter else 'Not initialized'}")
    print(f"  - LLM Provider: ✓ {'Available' if config.llm_provider != 'none' and config.llm_api_key else 'Not configured'}")
    print()
    
    print("=" * 80)
    if all_success:
        print("✅ All parsing strategies tested successfully!")
    else:
        print("⚠ Some strategies failed, but testing completed.")
    print("=" * 80)
    
    return all_success


if __name__ == "__main__":
    success = test_parsing_pipeline()
    sys.exit(0 if success else 1)
