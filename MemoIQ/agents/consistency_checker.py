"""Consistency checker agent."""

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
from MemoIQ.schema import ValidationRecord

logger = logging.getLogger(__name__)


def create_consistency_checker_agent(config: MemoIQConfig) -> Agent:
    """
    Create consistency checker agent.
    
    Args:
        config: MemoIQConfig
        
    Returns:
        Agent
    """
    return create_agent(
        name="consistency_checker",
        role="Data Consistency Validator",
        goal="Validate consistency across extracted fields",
        backstory="""You are a Consistency Checker agent specialized in validating consistency across extracted fields.
        You check for:
        - Date alignment (dates are consistent)
        - Number matching (numbers match across fields)
        - Logical consistency (values make sense together)
        - Cross-field validation
        You return validation results as JSON ValidationRecord objects.""",
        config=config,
    )


def check_consistency(
    extracted_fields: Dict[str, any],
    config: MemoIQConfig,
) -> list[ValidationRecord]:
    """
    Check consistency of extracted fields.
    
    Args:
        extracted_fields: Dict of field_id -> FieldExtraction
        config: MemoIQConfig
        
    Returns:
        List of ValidationRecord
    """
    fields_json = json.dumps(
        {k: {"value": v.value, "type": type(v.value).__name__} for k, v in extracted_fields.items()},
        indent=2,
        default=str,
    )
    
    prompt = f"""Check consistency of the following extracted fields:

{fields_json}

Return a JSON array of ValidationRecord objects:
[
    {{
        "field_id": "<field_id_or_null_for_global>",
        "status": "pass|warning|error",
        "message": "<validation_message>",
        "suggestions": ["<suggestion1>", "<suggestion2>"],
        "severity": <0-10>
    }}
]"""
    
    # Get response from LLM directly
    content = call_agent_llm_direct(config.agent_llm_config, prompt, temperature=0.7, max_tokens=2000)
    
    try:
        json_match = None
        if "```json" in content:
            json_match = content.split("```json")[1].split("```")[0].strip()
        elif "[" in content:
            json_start = content.find("[")
            json_end = content.rfind("]") + 1
            json_match = content[json_start:json_end]
        
        if json_match:
            data = json.loads(json_match)
            return [ValidationRecord(**item) for item in data]
        else:
            return []
    except Exception as e:
        logger.error(f"Error parsing consistency check: {e}", exc_info=True)
        return []
