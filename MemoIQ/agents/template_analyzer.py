"""Template analyzer agent."""

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
from MemoIQ.schema import TemplateSchema
from MemoIQ.template.template_parser import TemplateParser

logger = logging.getLogger(__name__)


def create_template_analyzer_agent(config: MemoIQConfig) -> Agent:
    """
    Create template analyzer agent.
    
    Args:
        config: MemoIQConfig
        
    Returns:
        Agent
    """
    return create_agent(
        name="template_analyzer",
        role="Template Analyzer",
        goal="Analyze template documents and extract field definitions accurately",
        backstory="""You are a Template Analyzer agent specialized in analyzing template documents and extracting field definitions.
        When given a template, you:
        1. Identify all fillable fields (empty cells, underscore lines, checkboxes)
        2. Infer field meanings from context (section headers, nearby labels)
        3. Create a structured TemplateSchema with field definitions
        You return your analysis as a JSON TemplateSchema object.""",
        config=config,
    )


def analyze_template(template_path: str, config: MemoIQConfig) -> TemplateSchema:
    """
    Analyze template and return TemplateSchema.
    
    Args:
        template_path: Path to template file
        config: MemoIQConfig
        
    Returns:
        TemplateSchema
    """
    parser = TemplateParser(config)
    template_schema = parser.parse_template(template_path)
    
    # Use LLM to refine field definitions if needed
    # For now, return the parsed schema directly
    return template_schema
