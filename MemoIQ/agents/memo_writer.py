"""Memo writer agent."""

import json
import logging
from typing import Dict

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
from MemoIQ.prompts.section_prompts import get_section_writing_prompt

logger = logging.getLogger(__name__)


def create_memo_writer_agent(config: MemoIQConfig) -> Agent:
    """
    Create memo writer agent.
    
    Args:
        config: MemoIQConfig
        
    Returns:
        Agent
    """
    return create_agent(
        name="memo_writer",
        role="Credit Memo Writer",
        goal="Write professional, well-structured narrative sections of credit memos",
        backstory="""You are a Memo Writer agent specialized in writing narrative sections of credit memos.
        When given extracted fields and context, you write professional, well-structured memo sections.
        You return your writing as a JSON object with section content and citations.
        You are clear, professional, and always include citations for factual claims.""",
        config=config,
    )


def write_section(
    section_name: str,
    extracted_fields: Dict[str, any],
    context: str,
    config: MemoIQConfig,
) -> Dict:
    """
    Write a memo section.
    
    Args:
        section_name: Name of section to write
        extracted_fields: Extracted field values
        context: Context from RAG retrieval
        config: MemoIQConfig
        
    Returns:
        Dict with section content and citations
    """
    prompt = get_section_writing_prompt(section_name, extracted_fields, context)
    
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
            return json.loads(json_match)
        else:
            return {"section_name": section_name, "content": "", "citations": []}
    except Exception as e:
        logger.error(f"Error parsing section writing: {e}", exc_info=True)
        return {"section_name": section_name, "content": "", "citations": []}
