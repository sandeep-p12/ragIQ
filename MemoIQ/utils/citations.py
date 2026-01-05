"""Citation utilities wrapping /src citation helpers."""

from typing import Any, Dict

from src.utils.io import format_citation


def format_citation_from_chunk(chunk: Dict[str, Any]) -> str:
    """
    Format citation from chunk dictionary.
    
    Args:
        chunk: Chunk dictionary with metadata
        
    Returns:
        Formatted citation string
    """
    return format_citation(chunk)


def format_citation_from_evidence(evidence: Dict[str, Any]) -> str:
    """
    Format citation from evidence pack entry.
    
    Args:
        evidence: Evidence pack entry with citation info
        
    Returns:
        Formatted citation string
    """
    # Extract citation info from evidence
    citation_parts = []
    
    if "doc_id" in evidence:
        citation_parts.append(f"Doc: {evidence['doc_id']}")
    
    if "page_span" in evidence:
        page_span = evidence["page_span"]
        if isinstance(page_span, (list, tuple)) and len(page_span) >= 2:
            if page_span[0] == page_span[1]:
                citation_parts.append(f"Page: {page_span[0]}")
            else:
                citation_parts.append(f"Pages: {page_span[0]}-{page_span[1]}")
    
    if "section_label" in evidence and evidence["section_label"]:
        citation_parts.append(f"Section: {evidence['section_label']}")
    
    return " | ".join(citation_parts) if citation_parts else "Unknown source"

