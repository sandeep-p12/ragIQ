"""Risk synthesizer agent."""

import json
import logging
from typing import Dict, List

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
from MemoIQ.prompts.section_prompts import get_risk_synthesis_prompt

logger = logging.getLogger(__name__)


def create_risk_synthesizer_agent(config: MemoIQConfig) -> Agent:
    """
    Create risk synthesizer agent.
    
    Args:
        config: MemoIQConfig
        
    Returns:
        Agent
    """
    return create_agent(
        name="risk_synthesizer",
        role="Risk Factor Synthesizer",
        goal="Synthesize risk factors from multiple document sources comprehensively",
        backstory="""You are a Risk Synthesizer agent specialized in synthesizing risk factors from multiple document sources.
        When given contexts from multiple documents, you identify and synthesize all significant risks.
        You return your synthesis as a JSON object with risks, severity levels, descriptions, and citations.
        You are comprehensive and prioritize risks by severity.""",
        config=config,
    )


def synthesize_risks(
    contexts: List[str],
    config: MemoIQConfig,
) -> Dict:
    """
    Synthesize risks from multiple contexts.
    
    Args:
        contexts: List of context strings from different documents
        config: MemoIQConfig
        
    Returns:
        Dict with synthesized risks
    """
    prompt = get_risk_synthesis_prompt(contexts)
    
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
            return {"risks": []}
    except Exception as e:
        logger.error(f"Error parsing risk synthesis: {e}", exc_info=True)
        return {"risks": []}
