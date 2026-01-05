"""Coordinator agent - workflow orchestrator."""

import logging
from typing import Any, Dict, List

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
from MemoIQ.schema import FieldExtraction, TemplateSchema, ValidationRecord

logger = logging.getLogger(__name__)


def create_coordinator_agent(config: MemoIQConfig) -> Agent:
    """
    Create coordinator agent.
    
    Args:
        config: MemoIQConfig
        
    Returns:
        Agent
    """
    return create_agent(
        name="coordinator",
        role="Workflow Coordinator",
        goal="Orchestrate the memo drafting workflow efficiently",
        backstory="""You are the Coordinator agent responsible for orchestrating the memo drafting workflow.
        Your workflow:
        1. Receive template schema from TemplateAnalyzer
        2. Create extraction plan (which fields, which agents)
        3. Assign tasks to specialist agents
        4. Collect structured results
        5. Trigger validations (ConsistencyChecker, PolicyChecker)
        6. Assemble final memo draft
        7. Request human review via HumanProxy
        8. Run revision pass on feedback
        You coordinate all other agents and manage the workflow state.""",
        config=config,
        allow_delegation=True,  # Coordinator can delegate tasks
    )


class WorkflowState:
    """Workflow state manager."""
    
    def __init__(self, template_schema: TemplateSchema):
        """Initialize workflow state."""
        self.template_schema = template_schema
        self.extracted_fields: Dict[str, FieldExtraction] = {}
        self.validation_records: List[ValidationRecord] = []
        self.extraction_plan: Dict[str, str] = {}  # field_id -> agent_name
        self.completed_tasks: List[str] = []
        self.pending_tasks: List[str] = []
    
    def create_extraction_plan(self) -> Dict[str, str]:
        """Create extraction plan mapping fields to agents."""
        plan = {}
        
        for field_id, field_def in self.template_schema.field_definitions.items():
            # Determine which agent should extract this field
            if field_def.type in ["currency", "number"] or "financial" in field_def.name.lower():
                plan[field_id] = "financial_extractor"
            elif "covenant" in field_def.name.lower() or field_def.type == "text":
                plan[field_id] = "covenant_extractor"
            elif field_def.field_type == "checkbox":
                plan[field_id] = "covenant_extractor"  # Use covenant extractor for checkboxes
            else:
                plan[field_id] = "covenant_extractor"  # Default
        
        self.extraction_plan = plan
        self.pending_tasks = list(plan.keys())
        return plan
    
    def mark_task_complete(self, field_id: str):
        """Mark a task as complete."""
        if field_id in self.pending_tasks:
            self.pending_tasks.remove(field_id)
        if field_id not in self.completed_tasks:
            self.completed_tasks.append(field_id)
    
    def add_extraction(self, field_id: str, extraction: FieldExtraction):
        """Add extracted field."""
        self.extracted_fields[field_id] = extraction
        self.mark_task_complete(field_id)
    
    def add_validation(self, record: ValidationRecord):
        """Add validation record."""
        self.validation_records.append(record)
    
    def is_complete(self) -> bool:
        """Check if all tasks are complete."""
        return len(self.pending_tasks) == 0
