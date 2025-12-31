"""UI utility functions for Streamlit."""

import logging
from typing import Any, Dict, Optional

import streamlit as st

logger = logging.getLogger(__name__)


def display_error(error: Exception, details: Optional[str] = None):
    """
    Display error in Streamlit UI.

    Args:
        error: Exception object
        details: Optional additional details
    """
    with st.expander("Error Details", expanded=True):
        st.error(str(error))
        if details:
            st.text(details)
        st.exception(error)


def format_progress_message(stage: str, progress: float) -> str:
    """
    Format progress message.

    Args:
        stage: Current stage
        progress: Progress (0-1)

    Returns:
        Formatted message
    """
    percentage = int(progress * 100)
    return f"{stage}: {percentage}%"


def format_stage_output(stage: str, output_data: Any) -> Dict[str, Any]:
    """
    Format output data for a processing stage.

    Args:
        stage: Stage name
        output_data: Output data from the stage

    Returns:
        Formatted dictionary for display
    """
    if stage == "file_loading":
        return {
            "File Name": output_data.get("file_name", "Unknown"),
            "File Size": f"{output_data.get('file_size', 0) / 1024:.2f} KB",
            "Page Count": output_data.get("page_count", 0),
            "File Type": output_data.get("file_type", "Unknown"),
        }
    elif stage == "strategy_selection":
        strategies = output_data.get("strategies", {})
        return {
            "Total Pages": output_data.get("total_pages", 0),
            "FAST Pages": sum(1 for s in strategies.values() if s == "FAST"),
            "HI_RES Pages": sum(1 for s in strategies.values() if s == "HI_RES"),
            "Page Strategies": strategies,
        }
    elif stage == "ocr_processing":
        return {
            "Pages Processed": output_data.get("pages_processed", 0),
            "Text Blocks Detected": output_data.get("text_blocks", 0),
            "Average Confidence": f"{output_data.get('avg_confidence', 0):.2f}",
            "OCR Results": output_data.get("results", {}),
        }
    elif stage == "layout_detection":
        counts = output_data.get("counts", {})
        detected_labels = output_data.get("detected_labels", {})
        model_classes = output_data.get("model_classes", {})
        
        result = {
            "Pages Processed": output_data.get("pages_processed", 0),
            "Titles Detected": counts.get("title", 0),
            "Text Blocks": counts.get("text", 0),
            "Tables": counts.get("table", 0),
            "Images": counts.get("image", 0),
            "Total Elements": sum(counts.values()),
        }
        
        # Add detected labels information
        if detected_labels:
            result["Detected Labels"] = detected_labels
        if model_classes:
            # Format model classes as a readable string
            classes_str = ", ".join([f"{id}: {name}" for id, name in sorted(model_classes.items())])
            result["Available Model Classes"] = classes_str
        
        return result
    elif stage == "post_processing":
        return {
            "Low Confidence Removed": output_data.get("low_confidence_removed", 0),
            "Duplicates Removed": output_data.get("duplicates_removed", 0),
            "Footnotes Associated": output_data.get("footnotes_associated", 0),
            "Overlaps Resolved": output_data.get("overlaps_resolved", 0),
        }
    elif stage == "reading_order":
        return {
            "Pages Processed": output_data.get("pages_processed", 0),
            "Blocks Sorted": output_data.get("blocks_sorted", 0),
            "Ordering Method": output_data.get("method", "XY-Cut"),
        }
    elif stage == "table_extraction":
        tables = output_data.get("tables", [])
        return {
            "Tables Extracted": len(tables),
            "Total Rows": sum(t.get("rows", 0) for t in tables),
            "Total Columns": sum(t.get("columns", 0) for t in tables),
            "Total Cells": sum(t.get("cells", 0) for t in tables),
            "Table Details": tables,
        }
    elif stage == "table_merging":
        return {
            "Tables Merged": output_data.get("tables_merged", 0),
            "Continuations Detected": output_data.get("continuations", 0),
            "Merge Operations": output_data.get("operations", []),
        }
    elif stage == "paragraph_splitting":
        return {
            "Paragraphs Created": output_data.get("paragraphs", 0),
            "Lists Detected": output_data.get("lists", 0),
            "Block Types": output_data.get("block_types", {}),
        }
    elif stage == "llm_formatting":
        return {
            "Tables Processed": output_data.get("tables_processed", 0),
            "Images Processed": output_data.get("images_processed", 0),
            "API Calls": output_data.get("api_calls", 0),
            "LLM Provider": output_data.get("provider", "none"),
        }
    elif stage == "markdown_generation":
        return {
            "Markdown Length": f"{output_data.get('length', 0)} characters",
            "Blocks Processed": output_data.get("blocks", 0),
            "Tables": output_data.get("tables", 0),
            "Images": output_data.get("images", 0),
        }
    else:
        # Generic formatting
        if isinstance(output_data, dict):
            return output_data
        return {"Output": str(output_data)}

