"""I/O utilities for JSONL and citation formatting."""

import json
from typing import Any, Dict, List


def load_jsonl(file_path: str) -> List[Dict[str, Any]]:
    """Load JSONL file into list of dictionaries.
    
    Args:
        file_path: Path to JSONL file
    
    Returns:
        List[Dict[str, Any]]: List of JSON objects
    """
    data = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def save_jsonl(data: List[Dict[str, Any]], file_path: str) -> None:
    """Save list of dictionaries to JSONL file.
    
    Args:
        data: List of dictionaries to save
        file_path: Path to output JSONL file
    """
    with open(file_path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def format_citation(chunk: Dict[str, Any]) -> str:
    """Format human-readable citation string from chunk.
    
    Args:
        chunk: Chunk dictionary with metadata
    
    Returns:
        str: Formatted citation string
    """
    doc_id = chunk.get("doc_id", "unknown")
    page_span = chunk.get("page_span", (0, 0))
    section_label = chunk.get("section_label", "Untitled")
    header_path = chunk.get("header_path")
    
    citation_parts = [f"Doc: {doc_id}"]
    
    if page_span[0] == page_span[1]:
        citation_parts.append(f"Page: {page_span[0]}")
    else:
        citation_parts.append(f"Pages: {page_span[0]}-{page_span[1]}")
    
    if header_path:
        citation_parts.append(f"Section: {header_path}")
    elif section_label:
        citation_parts.append(f"Section: {section_label}")
    
    # Add element-specific ranges if available
    metadata = chunk.get("metadata", {})
    if "row_range" in metadata:
        row_range = metadata["row_range"]
        citation_parts.append(f"Rows: {row_range[0]}-{row_range[1]}")
    if "item_range" in metadata:
        item_range = metadata["item_range"]
        citation_parts.append(f"Items: {item_range[0]}-{item_range[1]}")
    
    return " | ".join(citation_parts)

