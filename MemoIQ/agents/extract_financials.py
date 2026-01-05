"""Financial extractor agent."""

import json
import logging

try:
    from crewai import Agent
except ImportError as e:
    import sys
    error_msg = f"""crewai is required but not found. 

Install it with:
    python3 -m pip install "crewai>=0.80.0"

Current Python: {sys.executable}
Original error: {e}"""
    raise ImportError(error_msg) from e

from MemoIQ.agents.base import call_agent_llm_direct, create_agent
from MemoIQ.config import MemoIQConfig
from MemoIQ.prompts.field_prompts import get_financial_extraction_prompt
from MemoIQ.schema import Citation, FieldExtraction

logger = logging.getLogger(__name__)


def create_financial_extractor_agent(config: MemoIQConfig) -> Agent:
    """
    Create financial extractor agent.
    
    Args:
        config: MemoIQConfig
        
    Returns:
        Agent
    """
    return create_agent(
        name="financial_extractor",
        role="Financial Data Extractor",
        goal="Extract financial metrics and data from documents with precision and accuracy",
        backstory="""You are a Financial Extractor agent specialized in extracting financial metrics and data from documents.
        Your role is to extract financial values (amounts, ratios, percentages, etc.) from context.
        You return extractions as JSON FieldExtraction objects with value, confidence, and citations.
        You are precise with numbers and always include citations.""",
        config=config,
    )


def extract_financial_field(
    field_name: str,
    context: str,
    config: MemoIQConfig,
) -> FieldExtraction:
    """
    Extract financial field value.
    
    Args:
        field_name: Name of field to extract
        context: Context from RAG retrieval
        config: MemoIQConfig
        
    Returns:
        FieldExtraction
    """
    prompt = get_financial_extraction_prompt(field_name, context)
    
    # Get response from LLM directly
    content = call_agent_llm_direct(config.agent_llm_config, prompt, temperature=0.7, max_tokens=2000)
    
    # Parse JSON response
    try:
        # Extract JSON from response
        json_match = None
        if "```json" in content:
            json_match = content.split("```json")[1].split("```")[0].strip()
        elif "{" in content:
            json_start = content.find("{")
            json_end = content.rfind("}") + 1
            json_match = content[json_start:json_end]
        
        if json_match:
            data = json.loads(json_match)
            return FieldExtraction(**data)
        else:
            logger.warning(f"Could not parse JSON from response: {content}")
            return FieldExtraction(
                field_id=field_name.lower().replace(" ", "_"),
                value=None,
                confidence=0.0,
                citations=[],
            )
    except Exception as e:
        logger.error(f"Error parsing financial extraction: {e}", exc_info=True)
        return FieldExtraction(
            field_id=field_name.lower().replace(" ", "_"),
            value=None,
            confidence=0.0,
            citations=[],
        )
