"""OpenAI LLM provider implementation."""

import base64
import logging
from io import BytesIO
from typing import Optional

from openai import APIError, OpenAI, RateLimitError, AzureOpenAI
from azure.identity import DefaultAzureCredential, get_bearer_token_provider
from PIL import Image

from src.config.parsing import ParseForgeConfig
from src.core.interfaces import LLMProvider
from src.utils.env import (
    get_openai_api_key,
    get_azure_openai_api_key,
    get_azure_openai_endpoint,
    get_azure_openai_api_version,
)

logger = logging.getLogger(__name__)


class OpenAILLMProvider(LLMProvider):
    """OpenAI LLM provider implementation.
    
    Supports both OpenAI and Azure OpenAI.
    """
    
    def __init__(self, config: Optional[ParseForgeConfig] = None):
        """Initialize OpenAI LLM provider.
        
        Args:
            config: ParseForgeConfig with LLM settings (defaults to ParseForgeConfig())
        """
        self.config = config or ParseForgeConfig()
        self.client = None
        self.model_name = None
        
        if self.config.llm_provider == "openai":
            api_key = self.config.llm_api_key or get_openai_api_key()
            if api_key:
                self.client = OpenAI(api_key=api_key)
                self.model_name = self.config.llm_model
            else:
                logger.warning("OpenAI API key not found. LLM operations will be disabled.")
        elif self.config.llm_provider == "azure_openai":
            endpoint = self.config.llm_azure_endpoint or get_azure_openai_endpoint()
            
            if not endpoint:
                logger.warning("Azure OpenAI endpoint not found. LLM operations will be disabled.")
            else:
                # Use deployment name if provided, otherwise use model name
                deployment_name = self.config.llm_azure_deployment_name or self.config.llm_model
                api_version = self.config.llm_azure_api_version or get_azure_openai_api_version()
                
                # Use Azure AD authentication if enabled (default), otherwise use API key
                if self.config.llm_use_azure_ad:
                    # Azure AD authentication using DefaultAzureCredential
                    token_provider = get_bearer_token_provider(
                        DefaultAzureCredential(),
                        "https://cognitiveservices.azure.com/.default"
                    )
                    self.client = AzureOpenAI(
                        azure_endpoint=endpoint,
                        azure_ad_token_provider=token_provider,
                        api_version=api_version
                    )
                else:
                    # API key authentication (fallback)
                    api_key = self.config.llm_api_key or get_azure_openai_api_key()
                    if not api_key:
                        logger.warning("Azure OpenAI API key not found. Set AZURE_OPENAI_USE_AZURE_AD=true to use Azure AD authentication. LLM operations will be disabled.")
                    else:
                        self.client = OpenAI(
                            api_key=api_key,
                            api_version=api_version,
                            azure_endpoint=endpoint
                        )
                
                if self.client:
                    self.model_name = deployment_name
        elif self.config.llm_provider == "none":
            logger.info("LLM provider set to 'none'. LLM operations will be disabled.")
        else:
            logger.warning(f"LLM provider '{self.config.llm_provider}' not supported. Supported providers: 'openai', 'azure_openai', 'none'.")
    
    def generate(
        self,
        prompt: str,
        model: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: Optional[int] = None
    ) -> str:
        """Generate text from prompt.
        
        Args:
            prompt: Input prompt
            model: Model name (uses config default if not provided)
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate (uses config default if not provided)
            
        Returns:
            Generated text
            
        Raises:
            ValueError: If LLM client is not initialized
            APIError: If API call fails
        """
        if self.client is None:
            raise ValueError("LLM client not initialized. Check API key configuration.")
        
        model = model or self.model_name or self.config.llm_model
        max_tokens = max_tokens or self.config.llm_max_tokens
        
        try:
            response = self.client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return response.choices[0].message.content
        except RateLimitError as e:
            logger.error(f"Rate limit error: {e}")
            raise
        except APIError as e:
            logger.error(f"API error: {e}")
            raise
    
    def generate_vision(
        self,
        prompt: str,
        images: list,
        model: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: Optional[int] = None
    ) -> str:
        """Generate text from prompt with vision (images).
        
        Args:
            prompt: Input prompt
            images: List of image data (base64 strings or PIL Images)
            model: Model name (defaults to config model)
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            
        Returns:
            Generated text
        """
        if self.client is None:
            raise ValueError("LLM client not initialized. Check API key configuration.")
        
        model = model or self.model_name or self.config.llm_model
        max_tokens = max_tokens or self.config.llm_max_tokens
        
        # Convert images to format expected by OpenAI
        image_contents = []
        for img in images:
            if isinstance(img, str):
                # Assume base64 string
                image_contents.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{img}"}
                })
            elif hasattr(img, "save"):  # PIL Image
                # Convert PIL Image to base64
                buffered = BytesIO()
                img.save(buffered, format="PNG")
                img_str = base64.b64encode(buffered.getvalue()).decode()
                image_contents.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{img_str}"}
                })
            else:
                # Try to convert to PIL Image
                try:
                    if isinstance(img, bytes):
                        img = Image.open(BytesIO(img))
                    buffered = BytesIO()
                    img.save(buffered, format="PNG")
                    img_str = base64.b64encode(buffered.getvalue()).decode()
                    image_contents.append({
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{img_str}"}
                    })
                except Exception as e:
                    logger.warning(f"Failed to convert image to base64: {e}")
                    continue
        
        try:
            response = self.client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "user", "content": [
                        {"type": "text", "text": prompt},
                        *image_contents
                    ]}
                ],
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return response.choices[0].message.content
        except RateLimitError as e:
            logger.error(f"Rate limit error: {e}")
            raise
        except APIError as e:
            logger.error(f"API error: {e}")
            raise

