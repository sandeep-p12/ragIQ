"""Covenant extractor agent."""

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
from MemoIQ.prompts.field_prompts import get_covenant_extraction_prompt
from MemoIQ.schema import Citation, FieldExtraction

logger = logging.getLogger(__name__)


def create_covenant_extractor_agent(config: MemoIQConfig) -> Agent:
    """
    Create covenant extractor agent.
    
    Args:
        config: MemoIQConfig
        
    Returns:
        Agent
    """
    return create_agent(
        name="covenant_extractor",
        role="Covenant Terms Extractor",
        goal="Extract covenant terms and conditions from documents thoroughly and accurately",
        backstory="""You are a Covenant Extractor agent specialized in extracting covenant terms and conditions from documents.
        Your role is to extract relevant covenant information from context.
        You return extractions as JSON FieldExtraction objects with value, confidence, and citations.
        You are thorough and always include citations.""",
        config=config,
    )


def extract_covenant_field(
    field_name: str,
    context: str,
    config: MemoIQConfig,
) -> FieldExtraction:
    """
    Extract covenant field value.
    
    Args:
        field_name: Name of field to extract
        context: Context from RAG retrieval
        config: MemoIQConfig
        
    Returns:
        FieldExtraction
    """
    prompt = get_covenant_extraction_prompt(field_name, context)
    
    # Get response from LLM directly
    content = call_agent_llm_direct(config.agent_llm_config, prompt, temperature=0.7, max_tokens=2000)
    
    try:
        json_match = None
        if "```json" in content:
            json_match = content.split("```json")[1].split("```")[0].strip()
        elif "{" in content:
            json_start = content.find("{")
            json_end = content.rfind("}") + 1
            json_match = content[json_start:json_end]
        
        if json_match:
            data = json.loads(json_match)
            if "field_id" not in data:
                data["field_id"] = field_name.lower().replace(" ", "_")
            # Convert citation dicts to Citation objects if needed
            if "citations" in data and data["citations"]:
                citations_list = []
                for cit in data["citations"]:
                    if isinstance(cit, dict):
                        citations_list.append(Citation(**cit))
                    else:
                        citations_list.append(cit)
                data["citations"] = citations_list
            return FieldExtraction(**data)
        else:
            return FieldExtraction(
                field_id=field_name.lower().replace(" ", "_"),
                value=None,
                confidence=0.0,
                citations=[],
            )
    except Exception as e:
        logger.error(f"Error parsing covenant extraction: {e}", exc_info=True)
        return FieldExtraction(
            field_id=field_name.lower().replace(" ", "_"),
            value=None,
            confidence=0.0,
            citations=[],
        )
