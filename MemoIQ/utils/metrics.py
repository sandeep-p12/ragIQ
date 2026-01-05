"""Quality metrics for MemoIQ."""

from typing import Dict, List

from MemoIQ.schema import FieldExtraction, MemoDraft, ValidationRecord


def calculate_extraction_confidence(draft: MemoDraft) -> float:
    """
    Calculate average extraction confidence.
    
    Args:
        draft: MemoDraft
        
    Returns:
        Average confidence score (0.0-1.0)
    """
    if not draft.extracted_fields:
        return 0.0
    
    confidences = [field.confidence for field in draft.extracted_fields.values()]
    return sum(confidences) / len(confidences) if confidences else 0.0


def calculate_validation_score(draft: MemoDraft) -> Dict[str, any]:
    """
    Calculate validation score.
    
    Args:
        draft: MemoDraft
        
    Returns:
        Dict with validation metrics
    """
    total = len(draft.validation_report)
    if total == 0:
        return {
            "total": 0,
            "pass": 0,
            "warnings": 0,
            "errors": 0,
            "score": 1.0,
        }
    
    pass_count = sum(1 for v in draft.validation_report if v.status == "pass")
    warning_count = sum(1 for v in draft.validation_report if v.status == "warning")
    error_count = sum(1 for v in draft.validation_report if v.status == "error")
    
    # Score: pass = 1.0, warning = 0.5, error = 0.0
    score = (pass_count * 1.0 + warning_count * 0.5) / total
    
    return {
        "total": total,
        "pass": pass_count,
        "warnings": warning_count,
        "errors": error_count,
        "score": score,
    }


def calculate_completeness(draft: MemoDraft, template_schema) -> float:
    """
    Calculate field completeness.
    
    Args:
        draft: MemoDraft
        template_schema: TemplateSchema
        
    Returns:
        Completeness score (0.0-1.0)
    """
    total_fields = len(template_schema.field_definitions)
    if total_fields == 0:
        return 1.0
    
    filled_fields = sum(
        1 for field_id, field_def in template_schema.field_definitions.items()
        if field_id in draft.extracted_fields and draft.extracted_fields[field_id].value is not None
    )
    
    return filled_fields / total_fields if total_fields > 0 else 0.0

