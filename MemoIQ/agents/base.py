"""Base CrewAI configuration and agent setup."""

import logging
from typing import Any, Dict, Optional

try:
    from crewai import Agent, Task, Crew
except ImportError as e:
    import sys
    error_msg = f"""crewai is required but not found. 

Install it with:
    python3 -m pip install "crewai>=0.80.0"

Or if using a virtual environment:
    source .venv/bin/activate
    pip install "crewai>=0.80.0"

Current Python: {sys.executable}
Original error: {e}"""
    raise ImportError(error_msg) from e

from src.config.parsing import ParseForgeConfig
from src.providers.llm.openai_llm import OpenAILLMProvider
from MemoIQ.config import MemoIQConfig

logger = logging.getLogger(__name__)


def create_llm_provider(parse_config: ParseForgeConfig) -> OpenAILLMProvider:
    """
    Create LLM provider from ParseForgeConfig for CrewAI.
    
    Args:
        parse_config: ParseForgeConfig
        
    Returns:
        OpenAILLMProvider
    """
    if parse_config.llm_provider == "none" or not parse_config.llm_api_key:
        raise ValueError("LLM configuration required for agents")
    
    return OpenAILLMProvider(parse_config)


def create_agent(
    name: str,
    role: str,
    goal: str,
    backstory: str,
    config: MemoIQConfig,
    tools: Optional[list] = None,
    verbose: bool = True,
    allow_delegation: bool = False,
) -> Agent:
    """
    Create a CrewAI Agent with standard configuration.
    
    Args:
        name: Agent name
        role: Agent role
        goal: Agent goal
        backstory: Agent backstory
        config: MemoIQConfig
        tools: Optional list of tools
        verbose: Whether to be verbose
        allow_delegation: Whether agent can delegate tasks
        
    Returns:
        Agent
    """
    # Create LLM provider
    llm_provider = create_llm_provider(config.agent_llm_config)
    
    # Configure LLM for CrewAI
    # CrewAI can use langchain's LLM interface, but we'll set environment variables instead
    # to avoid dependency conflicts. CrewAI will pick up these from the environment.
    import os
    
    # Log the provider being used
    logger.debug(f"Creating agent '{name}' with LLM provider: {config.agent_llm_config.llm_provider}")
    
    # Set environment variables for CrewAI to use
    # Clear any existing Azure env vars first to avoid conflicts
    azure_env_vars = ["AZURE_OPENAI_ENDPOINT", "AZURE_OPENAI_API_VERSION", "AZURE_OPENAI_DEPLOYMENT_NAME", "AZURE_OPENAI_API_KEY"]
    for var in azure_env_vars:
        if var in os.environ:
            del os.environ[var]
    
    if config.agent_llm_config.llm_provider == "azure_openai":
        logger.debug(f"Configuring Azure OpenAI: endpoint={config.agent_llm_config.llm_azure_endpoint}, model={config.agent_llm_config.llm_model}")
        # Set Azure OpenAI environment variables
        if config.agent_llm_config.llm_azure_endpoint:
            os.environ["AZURE_OPENAI_ENDPOINT"] = config.agent_llm_config.llm_azure_endpoint
        if config.agent_llm_config.llm_azure_api_version:
            os.environ["AZURE_OPENAI_API_VERSION"] = config.agent_llm_config.llm_azure_api_version
        deployment_name = config.agent_llm_config.llm_azure_deployment_name or config.agent_llm_config.llm_model
        if deployment_name:
            os.environ["AZURE_OPENAI_DEPLOYMENT_NAME"] = deployment_name
        if config.agent_llm_config.llm_api_key:
            os.environ["AZURE_OPENAI_API_KEY"] = config.agent_llm_config.llm_api_key
        os.environ["OPENAI_MODEL_NAME"] = config.agent_llm_config.llm_model or "gpt-4o"
        # Clear OpenAI API key to avoid conflicts
        if "OPENAI_API_KEY" in os.environ:
            del os.environ["OPENAI_API_KEY"]
    elif config.agent_llm_config.llm_provider == "openai":
        # Set OpenAI environment variables
        if config.agent_llm_config.llm_api_key:
            os.environ["OPENAI_API_KEY"] = config.agent_llm_config.llm_api_key
        os.environ["OPENAI_MODEL_NAME"] = config.agent_llm_config.llm_model or "gpt-4o"
        logger.debug(f"Configuring OpenAI: model={config.agent_llm_config.llm_model}, api_key_set={bool(config.agent_llm_config.llm_api_key)}")
    else:
        # "none" provider - clear all LLM env vars
        if "OPENAI_API_KEY" in os.environ:
            del os.environ["OPENAI_API_KEY"]
        if "OPENAI_MODEL_NAME" in os.environ:
            del os.environ["OPENAI_MODEL_NAME"]
    
    os.environ["OPENAI_TEMPERATURE"] = str(config.agent_temperature)
    
    # Try to use langchain LLM if available, otherwise let CrewAI use defaults
    llm = None
    try:
        from langchain_openai import ChatOpenAI, AzureChatOpenAI
        if config.agent_llm_config.llm_provider == "azure_openai":
            llm = AzureChatOpenAI(
                azure_endpoint=config.agent_llm_config.llm_azure_endpoint,
                api_version=config.agent_llm_config.llm_azure_api_version,
                deployment_name=config.agent_llm_config.llm_azure_deployment_name or config.agent_llm_config.llm_model,
                openai_api_key=config.agent_llm_config.llm_api_key,
                temperature=config.agent_temperature,
                model_name=config.agent_llm_config.llm_model,
            )
        else:
            llm = ChatOpenAI(
                model=config.agent_llm_config.llm_model,
                temperature=config.agent_temperature,
                openai_api_key=config.agent_llm_config.llm_api_key,
            )
    except ImportError:
        # langchain-openai not available, CrewAI will use environment variables
        pass
    
    # Create agent - CrewAI will use environment variables if llm is None
    agent_kwargs = {
        "role": role,
        "goal": goal,
        "backstory": backstory,
        "verbose": verbose,
        "allow_delegation": allow_delegation,
        "tools": tools or [],
    }
    if llm is not None:
        agent_kwargs["llm"] = llm
    
    return Agent(**agent_kwargs)


def call_agent_llm_direct(
    config: ParseForgeConfig,
    prompt: str,
    temperature: float = 0.7,
    max_tokens: int = 2000,
) -> str:
    """
    Call LLM directly using ParseForgeConfig.
    
    Args:
        config: ParseForgeConfig
        prompt: Prompt string
        temperature: Sampling temperature
        max_tokens: Max tokens to generate
        
    Returns:
        Generated text
    """
    try:
        llm_provider = create_llm_provider(config)
        return llm_provider.generate(
            prompt,
            model=config.llm_model,
            temperature=temperature,
            max_tokens=max_tokens,
        )
    except Exception as e:
        logger.error(f"Error calling LLM: {e}", exc_info=True)
        return ""
