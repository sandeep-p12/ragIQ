"""Configuration management using Pydantic Settings for parsing."""

from pathlib import Path
from typing import Literal, Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class LLMConfig(BaseSettings):
    """LLM configuration."""

    provider: Literal["openai", "azure_openai", "none"] = "openai"
    model: str = "gpt-4o"
    api_key: Optional[str] = None
    # Azure OpenAI settings (optional)
    azure_endpoint: Optional[str] = None
    azure_api_version: str = "2025-01-01-preview"
    azure_deployment_name: Optional[str] = None
    use_azure_ad: bool = True  # Use Azure AD authentication (DefaultAzureCredential) instead of API key

    model_config = SettingsConfigDict(env_prefix="PARSEFORGE_LLM_", extra="ignore")


class StrategyConfig(BaseSettings):
    """Strategy configuration."""

    page_threshold: float = 0.6
    document_threshold: float = 0.2
    finance_page_threshold: float = 0.7
    finance_document_threshold: float = 0.15

    model_config = SettingsConfigDict(env_prefix="PARSEFORGE_", extra="ignore")


class ModelConfig(BaseSettings):
    """Model configuration."""

    model_dir: Path = Path("src/ai_models")
    yolo_layout_model: str = "doclayout_yolo_ft.pt"
    doctr_det_arch: str = "fast_base"
    doctr_reco_arch: str = "crnn_vgg16_bn"

    model_config = SettingsConfigDict(env_prefix="PARSEFORGE_", extra="ignore")


class ParseForgeConfig(BaseSettings):
    """Main ParseForge configuration."""

    # Device
    device: Literal["cpu", "cuda", "mps", "coreml"] = "cpu"

    # Processing
    batch_size: int = 50
    page_threshold: float = 0.6
    document_threshold: float = 0.2

    # Finance mode
    finance_mode: bool = False
    finance_page_threshold: float = 0.7
    finance_document_threshold: float = 0.15

    # Checkpoint
    checkpoint_dir: Path = Path("src/data/parsing/checkpoints")
    auto_resume: bool = True

    # Models - model_dir is relative to project root
    model_dir: Path = Path("src/ai_models")
    yolo_layout_model: str = "doclayout_yolo_ft.pt"
    doctr_det_arch: str = "fast_base"
    doctr_reco_arch: str = "crnn_vgg16_bn"

    # LLM
    llm_provider: Literal["openai", "azure_openai", "none"] = "openai"
    llm_model: str = "gpt-4o"
    llm_api_key: Optional[str] = None
    llm_max_tokens: int = 1000
    # Azure OpenAI settings (optional)
    llm_azure_endpoint: Optional[str] = None
    llm_azure_api_version: str = "2025-01-01-preview"
    llm_azure_deployment_name: Optional[str] = None
    llm_use_azure_ad: bool = True  # Use Azure AD authentication (DefaultAzureCredential) instead of API key

    # Streamlit
    streamlit_port: int = 8501
    streamlit_theme: Literal["light", "dark"] = "dark"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_prefix="PARSEFORGE_",
        case_sensitive=False,
        extra="ignore",
    )

    def get_llm_config(self) -> LLMConfig:
        """Get LLM configuration."""
        return LLMConfig(
            provider=self.llm_provider,
            model=self.llm_model,
            api_key=self.llm_api_key,
            azure_endpoint=self.llm_azure_endpoint,
            azure_api_version=self.llm_azure_api_version,
            azure_deployment_name=self.llm_azure_deployment_name,
            use_azure_ad=self.llm_use_azure_ad,
        )

    def get_strategy_config(self) -> StrategyConfig:
        """Get strategy configuration."""
        if self.finance_mode:
            return StrategyConfig(
                page_threshold=self.finance_page_threshold,
                document_threshold=self.finance_document_threshold,
            )
        return StrategyConfig(
            page_threshold=self.page_threshold,
            document_threshold=self.document_threshold,
        )

    def get_model_config(self) -> ModelConfig:
        """Get model configuration."""
        return ModelConfig(
            model_dir=self.model_dir,
            yolo_layout_model=self.yolo_layout_model,
            doctr_det_arch=self.doctr_det_arch,
            doctr_reco_arch=self.doctr_reco_arch,
        )

