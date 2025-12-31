"""Configuration dataclasses for retrieval subsystem."""

from dataclasses import dataclass
from typing import Optional

from src.utils.env import load_env


@dataclass
class EmbeddingConfig:
    """Configuration for OpenAI embeddings."""
    model: str = "text-embedding-3-small"
    batch_size: int = 100
    max_retries: int = 3
    # Azure OpenAI settings (optional)
    provider: str = "openai"  # "openai" or "azure_openai"
    api_key: Optional[str] = None
    azure_endpoint: Optional[str] = None
    azure_api_version: str = "2025-01-01-preview"
    azure_deployment_name: Optional[str] = None
    use_azure_ad: bool = True  # Use Azure AD authentication (DefaultAzureCredential) instead of API key
    
    @classmethod
    def from_env(cls, **overrides):
        """Create config from environment variables with optional overrides."""
        env = load_env()
        config = cls(**overrides)
        # Load provider preference
        if config.provider == "openai" and not overrides.get("provider"):
            config.provider = env.get("EMBEDDING_PROVIDER", "openai")
        # Load API key
        if not config.api_key:
            config.api_key = env.get("OPENAI_API_KEY") or env.get("AZURE_OPENAI_API_KEY")
        # Load Azure-specific settings if using Azure
        if config.provider == "azure_openai":
            if not config.azure_endpoint:
                config.azure_endpoint = env.get("AZURE_OPENAI_ENDPOINT")
            if not config.azure_deployment_name:
                config.azure_deployment_name = env.get("AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME") or env.get("AZURE_OPENAI_DEPLOYMENT_NAME")
            if not config.azure_api_version:
                config.azure_api_version = env.get("AZURE_OPENAI_API_VERSION", "2025-01-01-preview")
            # Check if Azure AD should be used (default True, can be overridden with env var)
            use_azure_ad_env = env.get("AZURE_OPENAI_USE_AZURE_AD", "true").lower()
            if use_azure_ad_env not in overrides:
                config.use_azure_ad = use_azure_ad_env in ("true", "1", "yes")
        return config


@dataclass
class PineconeConfig:
    """Configuration for Pinecone vector store."""
    api_key_env: str = "PINECONE_API_KEY"
    index_name_env: str = "PINECONE_INDEX_NAME"
    namespace: str = "children"
    top_k_dense: int = 300
    
    def __post_init__(self):
        """Load values from environment after initialization."""
        env = load_env()
        self.api_key = env.get(self.api_key_env)
        self.index_name = env.get(self.index_name_env, "hybrid-chunking")
        # Allow namespace override from env
        self.namespace = env.get("PINECONE_NAMESPACE", self.namespace)
    
    @classmethod
    def from_env(cls, **overrides):
        """Create config from environment variables with optional overrides."""
        config = cls(**overrides)
        config.__post_init__()
        return config


@dataclass
class RerankConfig:
    """Configuration for LLM reranking."""
    model: str = "gpt-4o"
    max_candidates_to_rerank: int = 50
    return_top_n: int = 15
    max_text_chars_per_candidate: int = 1200
    strict_json_output: bool = True
    
    @classmethod
    def from_env(cls, **overrides):
        """Create config from environment variables with optional overrides."""
        return cls(**overrides)


@dataclass
class RetrievalConfig:
    """Configuration for retrieval pipeline."""
    neighbor_same_page: int = 1
    neighbor_cross_page: int = 2
    include_parents: bool = True
    final_max_tokens: int = 12000
    min_primary_hits_to_keep: int = 3
    
    # Embedding config
    embedding_config: EmbeddingConfig = None
    
    # Pinecone config
    pinecone_config: PineconeConfig = None
    
    # Rerank config
    rerank_config: RerankConfig = None
    
    def __post_init__(self):
        """Initialize sub-configs if not provided."""
        if self.embedding_config is None:
            self.embedding_config = EmbeddingConfig.from_env()
        if self.pinecone_config is None:
            self.pinecone_config = PineconeConfig.from_env()
        if self.rerank_config is None:
            self.rerank_config = RerankConfig.from_env()
    
    @classmethod
    def from_env(cls, **overrides):
        """Create config from environment variables with optional overrides."""
        config = cls(**overrides)
        config.__post_init__()
        return config

