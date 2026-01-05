"""Human proxy agent for human-in-the-loop."""

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

from MemoIQ.agents.base import create_agent
from MemoIQ.config import MemoIQConfig

logger = logging.getLogger(__name__)


def create_human_proxy_agent(config: MemoIQConfig) -> Agent:
    """
    Create human proxy agent.
    
    Args:
        config: MemoIQConfig
        
    Returns:
        Agent
    """
    return create_agent(
        name="human_reviewer",
        role="Human Reviewer",
        goal="Review and provide feedback on memo drafts",
        backstory="""You are a Human Reviewer agent that represents human-in-the-loop review.
        You review memo drafts and provide feedback for revisions.
        You ensure quality and accuracy of the final memo.""",
        config=config,
    )
