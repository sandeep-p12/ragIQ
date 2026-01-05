"""MemoIQ configuration."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from src.config.chunking import ChunkConfig
from src.config.parsing import ParseForgeConfig
from src.config.parsing_strategies import StrategyEnum
from src.config.retrieval import RetrievalConfig


@dataclass
class MemoIQConfig:
    """Configuration for MemoIQ system."""
    
    # Run storage
    runs_dir: Path = field(default_factory=lambda: Path("MemoIQ/runs"))
    
    # LLM provider override (if None, uses .env defaults)
    llm_provider: Optional[str] = None
    
    # Agent LLM configuration (reuse ParseForgeConfig)
    agent_llm_config: Optional[ParseForgeConfig] = None
    
    # RAG configuration (reuse existing configs)
    parsing_config: Optional[ParseForgeConfig] = None
    chunking_config: Optional[ChunkConfig] = None
    retrieval_config: Optional[RetrievalConfig] = None
    
    # Parsing strategy
    parsing_strategy: Optional[StrategyEnum] = None
    
    # Agent-specific settings
    agent_temperature: float = 0.7
    agent_max_turns: int = 50
    agent_timeout: int = 300  # seconds
    
    def __post_init__(self):
        """Initialize configs from environment if not provided."""
        # Create ParseForgeConfig with provider override if specified
        parse_config_kwargs = {}
        if self.llm_provider is not None:
            parse_config_kwargs["llm_provider"] = self.llm_provider
        
        if self.agent_llm_config is None:
            self.agent_llm_config = ParseForgeConfig(**parse_config_kwargs)
        elif self.llm_provider is not None:
            # Override provider if config already exists
            self.agent_llm_config.llm_provider = self.llm_provider
            
        if self.parsing_config is None:
            self.parsing_config = ParseForgeConfig(**parse_config_kwargs)
        elif self.llm_provider is not None:
            # Override provider if config already exists
            self.parsing_config.llm_provider = self.llm_provider
            
        if self.chunking_config is None:
            self.chunking_config = ChunkConfig()
        if self.retrieval_config is None:
            self.retrieval_config = RetrievalConfig.from_env()
        
        # Ensure runs directory exists
        self.runs_dir.mkdir(parents=True, exist_ok=True)

