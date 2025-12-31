"""Centralized configuration for the RAG system."""

from src.config.chunking import ChunkConfig
from src.config.parsing import LLMConfig, ModelConfig, ParseForgeConfig, StrategyConfig
from src.config.parsing_strategies import (
    PageStrategy,
    StrategyEnum,
    determine_global_strategy,
    get_page_strategy,
)
from src.config.prompts import (
    BASE_LLM_PROMPT,
    IMAGE_DESCRIPTION_PROMPT,
    PAGE_PROCESSING_PROMPT,
    TABLE_EXTRACTION_PROMPT,
    TABLE_FORMATTING_PROMPT_TEMPLATE,
    get_table_formatting_prompt,
)
from src.config.retrieval import (
    EmbeddingConfig,
    PineconeConfig,
    RerankConfig,
    RetrievalConfig,
)

__all__ = [
    # Parsing config
    "ParseForgeConfig",
    "LLMConfig",
    "StrategyConfig",
    "ModelConfig",
    # Parsing strategies
    "StrategyEnum",
    "PageStrategy",
    "get_page_strategy",
    "determine_global_strategy",
    # Prompts
    "BASE_LLM_PROMPT",
    "IMAGE_DESCRIPTION_PROMPT",
    "PAGE_PROCESSING_PROMPT",
    "TABLE_EXTRACTION_PROMPT",
    "TABLE_FORMATTING_PROMPT_TEMPLATE",
    "get_table_formatting_prompt",
    # Chunking config
    "ChunkConfig",
    # Retrieval config
    "RetrievalConfig",
    "EmbeddingConfig",
    "PineconeConfig",
    "RerankConfig",
]
