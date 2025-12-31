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

