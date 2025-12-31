"""Environment variable loading utilities."""

import os
from typing import Dict, Optional

from dotenv import load_dotenv


def load_env() -> Dict[str, str]:
    """Load environment variables from .env file using python-dotenv.
    
    Returns:
        Dict[str, str]: Dictionary of environment variables
    """
    load_dotenv()
    return dict(os.environ)


def get_openai_api_key() -> Optional[str]:
    """Get OPENAI_API_KEY from .env.
    
    Returns:
        Optional[str]: OpenAI API key if found, None otherwise
    """
    load_dotenv()
    return os.getenv("OPENAI_API_KEY")


def get_azure_openai_api_key() -> Optional[str]:
    """Get AZURE_OPENAI_API_KEY from .env.
    
    Returns:
        Optional[str]: Azure OpenAI API key if found, None otherwise
    """
    load_dotenv()
    return os.getenv("AZURE_OPENAI_API_KEY")


def get_azure_openai_endpoint() -> Optional[str]:
    """Get AZURE_OPENAI_ENDPOINT from .env.
    
    Returns:
        Optional[str]: Azure OpenAI endpoint if found, None otherwise
    """
    load_dotenv()
    return os.getenv("AZURE_OPENAI_ENDPOINT")


def get_azure_openai_api_version() -> str:
    """Get AZURE_OPENAI_API_VERSION from .env.
    
    Returns:
        str: Azure OpenAI API version (defaults to "2025-01-01-preview")
    """
    load_dotenv()
    return os.getenv("AZURE_OPENAI_API_VERSION", "2025-01-01-preview")

