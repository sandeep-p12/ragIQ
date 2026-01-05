"""Draft comparison and delta log generation."""

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

from MemoIQ.schema import MemoDraft


def compare_drafts(draft_v1: MemoDraft, draft_v2: MemoDraft) -> Dict[str, Any]:
    """
    Compare two drafts and generate delta log.
    
    Args:
        draft_v1: First draft
        draft_v2: Second draft (revised)
        
    Returns:
        Delta log dictionary
    """
    delta = {
        "timestamp": datetime.now().isoformat(),
        "from_version": draft_v1.draft_version,
        "to_version": draft_v2.draft_version,
        "changed_fields": [],
        "added_fields": [],
        "removed_fields": [],
        "validation_changes": [],
    }
    
    # Compare extracted fields
    v1_fields = set(draft_v1.extracted_fields.keys())
    v2_fields = set(draft_v2.extracted_fields.keys())
    
    # Added fields
    for field_id in v2_fields - v1_fields:
        delta["added_fields"].append({
            "field_id": field_id,
            "value": draft_v2.extracted_fields[field_id].value,
        })
    
    # Removed fields
    for field_id in v1_fields - v2_fields:
        delta["removed_fields"].append({
            "field_id": field_id,
            "old_value": draft_v1.extracted_fields[field_id].value,
        })
    
    # Changed fields
    for field_id in v1_fields & v2_fields:
        v1_field = draft_v1.extracted_fields[field_id]
        v2_field = draft_v2.extracted_fields[field_id]
        
        if v1_field.value != v2_field.value:
            delta["changed_fields"].append({
                "field_id": field_id,
                "old_value": v1_field.value,
                "new_value": v2_field.value,
                "confidence_change": v2_field.confidence - v1_field.confidence,
            })
    
    # Compare validation reports
    v1_validation_ids = {v.field_id for v in draft_v1.validation_report if v.field_id}
    v2_validation_ids = {v.field_id for v in draft_v2.validation_report if v.field_id}
    
    for field_id in v1_validation_ids | v2_validation_ids:
        v1_records = [v for v in draft_v1.validation_report if v.field_id == field_id]
        v2_records = [v for v in draft_v2.validation_report if v.field_id == field_id]
        
        if v1_records != v2_records:
            delta["validation_changes"].append({
                "field_id": field_id,
                "old_status": v1_records[0].status if v1_records else None,
                "new_status": v2_records[0].status if v2_records else None,
            })
    
    return delta


def save_delta_log(run_dir: Path, delta: Dict[str, Any], version: int) -> Path:
    """Save delta log to run directory."""
    file_path = run_dir / "outputs" / f"delta_log_v{version}.json"
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(delta, f, indent=2, default=str, ensure_ascii=False)
    return file_path

