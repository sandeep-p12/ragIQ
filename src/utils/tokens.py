"""Token counting and structure confidence utilities."""

from typing import List

import tiktoken


def count_tokens(text: str, model: str = "gpt-3.5-turbo") -> int:
    """Count tokens in text using tiktoken.
    
    Args:
        text: Text to count tokens for
        model: Model name for tokenizer (default: gpt-3.5-turbo)
    
    Returns:
        int: Number of tokens
    """
    try:
        encoding = tiktoken.encoding_for_model(model)
        return len(encoding.encode(text))
    except KeyError:
        # Fallback to cl100k_base if model not found
        encoding = tiktoken.get_encoding("cl100k_base")
        return len(encoding.encode(text))


def compute_structure_confidence(
    repair_records: List[dict], 
    page_content: str
) -> float:
    """Compute heuristic structure confidence score (0.0-1.0).
    
    Args:
        repair_records: List of repair records applied to the page
        page_content: Page content string
    
    Returns:
        float: Confidence score between 0.0 and 1.0
    """
    if not repair_records:
        return 1.0
    
    # Base confidence starts at 1.0
    confidence = 1.0
    
    # Penalize based on number of repairs
    repair_count = len(repair_records)
    confidence -= min(0.3, repair_count * 0.1)
    
    # Penalize based on repair types (table repairs are more serious)
    for record in repair_records:
        repair_type = record.get("repair_type", "")
        if repair_type == "table_repair":
            confidence -= 0.15
        elif repair_type == "list_repair":
            confidence -= 0.1
        elif repair_type == "section_repair":
            confidence -= 0.05
    
    # Boost confidence if content is well-structured (has headings, proper formatting)
    has_headings = any(line.strip().startswith("#") for line in page_content.split("\n"))
    if has_headings:
        confidence += 0.1
    
    # Ensure confidence is between 0.0 and 1.0
    return max(0.0, min(1.0, confidence))

