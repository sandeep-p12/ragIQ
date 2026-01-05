"""Policy checker agent."""

import json
import logging
from pathlib import Path
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


def create_policy_checker_agent(config: MemoIQConfig) -> Agent:
    """
    Create policy checker agent.
    
    Args:
        config: MemoIQConfig
        
    Returns:
        Agent
    """
    return create_agent(
        name="policy_checker",
        role="Credit Policy Validator",
        goal="Validate extracted fields against credit policy rules",
        backstory="""You are a Policy Checker agent specialized in validating extracted fields against credit policy rules.
        You check fields against policy rules (max/min values, ranges, required fields, etc.).
        You return validation results as JSON ValidationRecord objects.""",
        config=config,
    )


def check_policy(
    extracted_fields: Dict[str, any],
    policy_rules_path: str,
    config: MemoIQConfig,
) -> list[ValidationRecord]:
    """
    Check extracted fields against policy rules.
    
    Args:
        extracted_fields: Dict of field_id -> FieldExtraction
        policy_rules_path: Path to policy rules JSON file
        config: MemoIQConfig
        
    Returns:
        List of ValidationRecord
    """
    # Load policy rules (optional - if file doesn't exist or is empty, skip validation)
    policy_rules = {}
    rules_list = []
    
    if Path(policy_rules_path).exists():
        try:
            with open(policy_rules_path, 'r') as f:
                policy_rules = json.load(f)
            rules_list = policy_rules.get("rules", [])
        except Exception as e:
            logger.warning(f"Error loading policy rules from {policy_rules_path}: {e}")
            rules_list = []
    
    # If no policy rules, return empty validation list (no validation needed)
    if not rules_list:
        logger.info("No policy rules found or rules list is empty. Skipping policy validation.")
        return []
    
    logger.info(f"Validating {len(extracted_fields)} fields against {len(rules_list)} policy rules")
    
    fields_json = json.dumps(
        {k: {"value": v.value} for k, v in extracted_fields.items()},
        indent=2,
        default=str,
    )
    
    rules_json = json.dumps(rules_list, indent=2)
    
    prompt = f"""Check the following extracted fields against policy rules:

Fields:
{fields_json}

Policy Rules:
{rules_json}

Return a JSON array of ValidationRecord objects. Each record must have:
- field_id: string (or null for global validations)
- status: ONE of "pass", "warning", or "error" (NOT multiple values)
- message: string describing the validation result
- suggestions: array of strings (optional)
- severity: number 0-10 (0=info, 5=warning, 10=error)

Example:
[
    {{
        "field_id": "loan_amount",
        "status": "error",
        "message": "Loan amount exceeds maximum policy limit",
        "suggestions": ["Reduce loan amount to $500,000"],
        "severity": 10
    }}
]

IMPORTANT: The "status" field must be exactly one of: "pass", "warning", or "error". Do NOT use multiple values like "pass|error"."""
    
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
            # Normalize status values - handle cases where LLM returns invalid status
            normalized_records = []
            for item in data:
                # Normalize status field
                status = item.get("status", "warning")
                if isinstance(status, str):
                    # Handle cases like "pass|error" by taking the most severe
                    if "|" in status:
                        statuses = [s.strip() for s in status.split("|")]
                        # Priority: error > warning > pass
                        if "error" in statuses:
                            status = "error"
                        elif "warning" in statuses:
                            status = "warning"
                        else:
                            status = "pass"
                        logger.warning(f"Normalized status '{item.get('status')}' to '{status}'")
                    
                    # Ensure it's a valid status
                    if status not in ["pass", "warning", "error"]:
                        logger.warning(f"Invalid status '{status}', defaulting to 'warning'")
                        status = "warning"
                
                item["status"] = status
                normalized_records.append(ValidationRecord(**item))
            
            return normalized_records
        else:
            return []
    except Exception as e:
        logger.error(f"Error parsing policy check: {e}", exc_info=True)
        return []
