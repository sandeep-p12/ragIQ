"""Test script for MemoIQ workflow with LLM_FULL strategy and OpenAI config."""

import hashlib
import logging
import sys
from pathlib import Path

# Add project root to path
_project_root = Path(__file__).parent.parent.resolve()
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from src.config.parsing import ParseForgeConfig
from src.config.parsing_strategies import StrategyEnum
from src.config.retrieval import RetrievalConfig, EmbeddingConfig
from src.utils.env import get_openai_api_key, load_env

from MemoIQ.agents.orchestrator import MemoIQOrchestrator
from MemoIQ.config import MemoIQConfig
# Don't import parse_to_markdown directly - we'll use it through the module
from MemoIQ import rag

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_with_cache(
    file_path: str,
    config: ParseForgeConfig,
    strategy: StrategyEnum,
    cache_dir: Path,
) -> str:
    """
    Parse document to markdown with caching support for LLM_FULL strategy.
    This is only for testing purposes to avoid expensive LLM calls.
    
    Args:
        file_path: Path to document file
        config: ParseForge configuration
        strategy: Parsing strategy
        cache_dir: Cache directory path
        
    Returns:
        Markdown string
    """
    # Only cache for LLM_FULL strategy
    # Use rag_adapter module to get the current function (which may be monkey-patched)
    from MemoIQ.rag import rag_adapter
    current_parse_func = rag_adapter.parse_to_markdown
    
    if strategy != StrategyEnum.LLM_FULL:
        return current_parse_func(file_path, config, strategy=strategy)
    
    # Setup cache directory
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate cache key based on file path, modification time, and strategy
    file_path_obj = Path(file_path).resolve()  # Use absolute path for consistency
    if not file_path_obj.exists():
        logger.warning(f"File not found: {file_path}, cannot use cache")
        return current_parse_func(file_path, config, strategy=strategy)
    
    file_mtime = file_path_obj.stat().st_mtime
    # Use absolute path in cache key for consistency
    cache_key_data = f"{str(file_path_obj)}_{file_mtime}_{strategy}"
    # Import hashlib inside function to ensure it's available when called from monkey-patched context
    import hashlib as _h
    cache_key = _h.md5(cache_key_data.encode()).hexdigest()
    cache_file = cache_dir / f"{cache_key}.md"
    
    # Check if cached markdown exists
    if cache_file.exists():
        logger.info(f"📦 Loading cached LLM markdown from {cache_file.name}")
        try:
            with open(cache_file, 'r', encoding='utf-8') as f:
                cached_markdown = f.read()
            
            if cached_markdown:
                logger.info(f"✓ Using cached markdown ({len(cached_markdown)} chars) - skipping expensive LLM call")
                return cached_markdown
        except Exception as e:
            logger.warning(f"Failed to load cached markdown: {e}, will regenerate")
    
    # Parse document (this will call LLM)
    # Get the original function (stored when we monkey-patched) to avoid recursion
    from MemoIQ.rag import rag_adapter
    original_parse = getattr(rag_adapter, '_original_parse_to_markdown', None)
    if original_parse is None:
        # This should not happen - monkey-patch should have stored the original
        # Import directly from source to get the true original (before any patches)
        logger.warning("⚠️  _original_parse_to_markdown not found! Importing original directly.")
        from MemoIQ.rag.rag_adapter import parse_to_markdown as _direct_import
        # But wait - this will also be the patched version if module was already imported
        # So we need to get it from the actual function object
        import inspect
        import MemoIQ.rag.rag_adapter as _module
        # Get the function from the module's __dict__ before any patches
        # Actually, the safest is to just raise an error
        raise RuntimeError(
            "_original_parse_to_markdown not stored! "
            "Monkey-patch must store original before replacing function."
        )
    
    logger.info(f"🔄 No cache found, parsing with LLM (this may be expensive)...")
    logger.info(f"   File: {file_path}")
    logger.info(f"   Strategy: {strategy}")
    logger.info(f"   This may take several minutes...")
    markdown = original_parse(file_path, config, strategy=strategy)
    
    # Save markdown to cache if parsing was successful
    if markdown:
        try:
            logger.info(f"💾 Saving LLM markdown to cache: {cache_file.name}")
            with open(cache_file, 'w', encoding='utf-8') as f:
                f.write(markdown)
            logger.info(f"✓ Cached markdown saved ({len(markdown)} chars)")
            logger.info(f"   Cache file: {cache_file}")
        except Exception as e:
            logger.warning(f"Failed to save cached markdown: {e}")
    else:
        logger.warning("⚠️  Parsing returned empty markdown, not caching")
    
    return markdown


def test_memoiq_workflow():
    """Test MemoIQ workflow with LLM_FULL strategy and OpenAI config."""
    
    logger.info("=" * 80)
    logger.info("Testing MemoIQ Workflow")
    logger.info("=" * 80)
    
    # Step 1: Check for test files
    project_root = Path(__file__).parent.parent.resolve()
    template_path = project_root / "Exhibit-B.-Credit-Memo-Template.pdf"
    reference_path = project_root / "Sample-Enhanced-Memo.pdf"
    
    if not template_path.exists():
        logger.error(f"Template file not found: {template_path}")
        logger.info("Please ensure Exhibit-B.-Credit-Memo-Template.pdf exists in project root")
        return False
    
    if not reference_path.exists():
        logger.error(f"Reference file not found: {reference_path}")
        logger.info("Please ensure Sample-Enhanced-Memo.pdf exists in project root")
        return False
    
    # Use the provided template and reference document
    reference_doc_paths = [str(reference_path)]
    
    logger.info(f"Template: {template_path}")
    logger.info(f"Reference docs: {reference_doc_paths}")
    
    # Check cache directory
    cache_dir = project_root / "MemoIQ" / "cache" / "llm_markdown"
    if cache_dir.exists():
        cache_files = list(cache_dir.glob("*.md"))
        logger.info(f"Found {len(cache_files)} cached markdown files in {cache_dir}")
    else:
        logger.info(f"Cache directory will be created at: {cache_dir}")
    
    # Step 2: Load environment and check API key
    env = load_env()
    openai_api_key = get_openai_api_key()
    
    if not openai_api_key:
        logger.error("OPENAI_API_KEY not found in environment")
        logger.info("Please set OPENAI_API_KEY in .env file")
        return False
    
    logger.info(f"✓ OpenAI API key found: {openai_api_key[:10]}...")
    
    # Step 3: Create ParseForgeConfig with OpenAI and LLM_FULL strategy
    logger.info("\n" + "=" * 80)
    logger.info("Step 1: Creating ParseForgeConfig with OpenAI provider")
    logger.info("=" * 80)
    
    parsing_config = ParseForgeConfig(
        llm_provider="openai",
        llm_api_key=openai_api_key,
        llm_model=env.get("PARSEFORGE_LLM_MODEL") or env.get("OPENAI_MODEL_NAME") or "gpt-4o",
        # Explicitly set Azure settings to None
        llm_azure_endpoint=None,
        llm_azure_api_version="2025-01-01-preview",
        llm_azure_deployment_name=None,
        llm_use_azure_ad=False,
        # Parsing settings
        device="cpu",
        batch_size=50,
        page_threshold=0.6,
        document_threshold=0.2,
        auto_resume=True,
    )
    
    logger.info(f"✓ ParseForgeConfig created:")
    logger.info(f"  - LLM Provider: {parsing_config.llm_provider}")
    logger.info(f"  - LLM Model: {parsing_config.llm_model}")
    logger.info(f"  - API Key set: {bool(parsing_config.llm_api_key)}")
    logger.info(f"  - Azure Endpoint: {parsing_config.llm_azure_endpoint}")
    
    # Step 4: Create MemoIQConfig
    logger.info("\n" + "=" * 80)
    logger.info("Step 2: Creating MemoIQConfig")
    logger.info("=" * 80)
    
    # Create retrieval config with OpenAI provider (not Azure)
    from src.config.retrieval import RetrievalConfig, EmbeddingConfig
    embedding_config = EmbeddingConfig(
        provider="openai",
        api_key=parsing_config.llm_api_key,
        azure_endpoint=None,
        azure_deployment_name=None,
        azure_api_version="2025-01-01-preview",
        use_azure_ad=False,  # CRITICAL: Disable Azure AD
    )
    retrieval_config = RetrievalConfig(
        embedding_config=embedding_config
    )
    
    config = MemoIQConfig(
        llm_provider="openai",
        parsing_config=parsing_config,
        agent_llm_config=parsing_config,  # Use same config for agents
        retrieval_config=retrieval_config,  # Use OpenAI embedding config
        parsing_strategy=StrategyEnum.LLM_FULL,  # Use LLM_FULL strategy
    )
    
    logger.info(f"✓ MemoIQConfig created:")
    logger.info(f"  - LLM Provider: {config.llm_provider}")
    logger.info(f"  - Parsing Strategy: {config.parsing_strategy}")
    logger.info(f"  - Agent LLM Config Provider: {config.agent_llm_config.llm_provider}")
    
    # Step 3: Apply caching monkey-patch BEFORE creating orchestrator
    # This ensures all parsing uses cache for LLM_FULL strategy
    # Must be done before orchestrator imports doc_ingest
    logger.info("\n" + "=" * 80)
    logger.info("Step 3: Setting up caching for LLM_FULL strategy")
    logger.info("=" * 80)
    
    from MemoIQ.rag import rag_adapter
    
    # Store original function - we'll need it in parse_with_cache to avoid recursion
    original_parse = rag_adapter.parse_to_markdown
    # Store it as an attribute so parse_with_cache can access it
    rag_adapter._original_parse_to_markdown = original_parse
    
    # Replace with cached version for LLM_FULL strategy
    def cached_parse_to_markdown(file_path, config, generate_image_descriptions=True, strategy=None):
        file_name = Path(file_path).name
        if strategy == StrategyEnum.LLM_FULL:
            logger.info(f"🔧 [CACHE] Parsing {file_name} with LLM_FULL strategy (will use cache if available)")
            return parse_with_cache(file_path, config, strategy, cache_dir)
        else:
            # For non-LLM_FULL strategies (like AUTO for template), use original
            logger.info(f"🔧 [NO-CACHE] Parsing {file_name} with strategy {strategy} (no caching)")
            return original_parse(file_path, config, generate_image_descriptions, strategy)
    
    logger.info(f"🔧 Monkey-patching rag_adapter.parse_to_markdown with caching support")
    rag_adapter.parse_to_markdown = cached_parse_to_markdown
    
    # Also patch it in doc_ingest module BEFORE it gets imported by orchestrator
    from MemoIQ.rag import doc_ingest
    doc_ingest.parse_to_markdown = cached_parse_to_markdown
    
    logger.info(f"✅ Monkey-patch applied - LLM_FULL parsing will use cache")
    logger.info(f"   Cache directory: {cache_dir}")
    logger.info(f"   Patched in: rag_adapter and doc_ingest modules")
    
    # Step 4: Create orchestrator (after patching, so it sees the patched function)
    logger.info("\n" + "=" * 80)
    logger.info("Step 4: Creating MemoIQOrchestrator")
    logger.info("=" * 80)
    
    try:
        orchestrator = MemoIQOrchestrator(config)
        logger.info("✓ Orchestrator created successfully")
    except Exception as e:
        logger.error(f"✗ Failed to create orchestrator: {e}", exc_info=True)
        return False
    
    # Step 5: Test agent creation
    logger.info("\n" + "=" * 80)
    logger.info("Step 5: Testing agent creation")
    logger.info("=" * 80)
    
    try:
        from MemoIQ.agents.base import create_agent
        
        test_agent = create_agent(
            name="test_agent",
            role="Test Agent",
            goal="Test goal",
            backstory="Test backstory",
            config=config,
        )
        logger.info("✓ Agent created successfully")
        logger.info(f"  - Agent role: {test_agent.role}")
        logger.info(f"  - Agent goal: {test_agent.goal}")
    except Exception as e:
        logger.error(f"✗ Failed to create agent: {e}", exc_info=True)
        return False
    
    # Step 6: Test full workflow with caching
    logger.info("\n" + "=" * 80)
    logger.info("Step 6: Running full MemoIQ workflow with LLM_FULL strategy")
    logger.info("=" * 80)
    logger.info("STRICT ENFORCEMENT: All parsing will use LLM_FULL strategy")
    logger.info("  - TEMPLATE parsing: Uses LLM_FULL (from config.parsing_strategy)")
    logger.info("  - REFERENCE documents: Use LLM_FULL (from config.parsing_strategy)")
    logger.info("  → LLM_FULL bypasses: layout detection, OCR, table extraction")
    logger.info("  → Will use cached markdown if available")
    logger.info("  → NO table extraction, OCR, or layout detection logs should appear")
    
    try:
        logger.info("Running full MemoIQ workflow...")
        results = orchestrator.run(
            template_path=str(template_path),
            reference_doc_paths=reference_doc_paths,
        )
            
        logger.info("✓ Full workflow completed successfully")
        logger.info(f"  - Run directory: {results['run_dir']}")
        logger.info(f"  - Draft path: {results['memo_draft'].draft_docx_path}")
        logger.info(f"  - Extracted fields: {len(results['workflow_state'].extracted_fields)}")
        logger.info(f"  - Validation records: {len(results['workflow_state'].validation_records)}")
        
        # Verify cache was created
        cache_files = list(cache_dir.glob("*.md"))
        if cache_files:
            logger.info(f"✓ Cache files created: {len(cache_files)} files")
            for cache_file in cache_files:
                size = cache_file.stat().st_size
                logger.info(f"  - {cache_file.name}: {size:,} bytes")
        else:
            logger.warning("⚠️  No cache files found - markdown may not have been cached")
        
    except Exception as e:
        logger.error(f"✗ Full workflow failed: {e}", exc_info=True)
        return False
    finally:
        # Restore original function in both modules
        rag_adapter.parse_to_markdown = original_parse
        from MemoIQ.rag import doc_ingest
        doc_ingest.parse_to_markdown = original_parse
        logger.info("✓ Restored original parse_to_markdown function in all modules")
    
    logger.info("\n" + "=" * 80)
    logger.info("Test Summary")
    logger.info("=" * 80)
    logger.info("✓ Configuration setup: PASSED")
    logger.info("✓ Agent creation: PASSED")
    logger.info("✓ Caching setup: PASSED")
    logger.info("✓ Full workflow: PASSED")
    logger.info("✓ All core components are working!")
    logger.info("")
    logger.info("Note: Markdown cache saved for LLM_FULL strategy")
    logger.info(f"      Cache directory: {cache_dir}")
    logger.info("      Next run will reuse cached markdown (faster)")
    
    return True


if __name__ == "__main__":
    success = test_memoiq_workflow()
    sys.exit(0 if success else 1)

