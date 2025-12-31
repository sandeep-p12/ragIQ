"""LlamaIndex integration: MarkdownNodeParser + HierarchicalNodeParser with page awareness."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from llama_index.core import Document
from llama_index.core.node_parser import MarkdownNodeParser, HierarchicalNodeParser
from llama_index.core.schema import BaseNode, NodeWithScore

from src.schema.chunk import PageBlock


def create_markdown_parser() -> MarkdownNodeParser:
    """Create and configure MarkdownNodeParser.
    
    Returns:
        MarkdownNodeParser: Configured parser
    """
    return MarkdownNodeParser()


def create_hierarchical_parser(
    chunk_sizes: Optional[List[int]] = None,
    chunk_overlap: Optional[int] = None
) -> HierarchicalNodeParser:
    """Create and configure HierarchicalNodeParser.
    
    Args:
        chunk_sizes: List of chunk sizes for hierarchy (default: [2048, 512, 128])
        chunk_overlap: Overlap between chunks (default: 20)
    
    Returns:
        HierarchicalNodeParser: Configured parser
    """
    if chunk_sizes is None:
        chunk_sizes = [2048, 512, 128]
    if chunk_overlap is None:
        chunk_overlap = 20
    
    # Use from_defaults which handles node_parser_map internally
    try:
        return HierarchicalNodeParser.from_defaults(
            chunk_sizes=chunk_sizes,
            chunk_overlap=chunk_overlap
        )
    except Exception:
        # Fallback: create with explicit node_parser_map
        from llama_index.core.node_parser import SimpleNodeParser
        from llama_index.core.text_splitter import TokenTextSplitter
        
        node_parser_map = {}
        for chunk_size in chunk_sizes:
            text_splitter = TokenTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap
            )
            # Create a simple parser for each chunk size
            node_parser_map[chunk_size] = SimpleNodeParser(
                text_splitter=text_splitter
            )
        
        return HierarchicalNodeParser(
            chunk_sizes=chunk_sizes,
            chunk_overlap=chunk_overlap,
            node_parser_map=node_parser_map
        )


def parse_with_llamaindex(
    page_block: PageBlock,
    use_hierarchical: bool = True
) -> List[BaseNode]:
    """Parse page block with LlamaIndex.
    
    Uses MarkdownNodeParser, optionally with HierarchicalNodeParser.
    Post-processes nodes to attach page metadata.
    
    Args:
        page_block: PageBlock to parse
        use_hierarchical: Whether to use HierarchicalNodeParser
    
    Returns:
        List[BaseNode]: Parsed nodes with page metadata
    """
    # Create document from page block
    doc = Document(
        text=page_block.content,
        metadata={
            "page_no": page_block.page_no,
            "page_span": page_block.page_span,
            "page_nos": page_block.page_nos,
            "structure_confidence": page_block.structure_confidence,
        }
    )
    
    # Parse with MarkdownNodeParser first
    markdown_parser = create_markdown_parser()
    nodes = markdown_parser.get_nodes_from_documents([doc])
    
    # Optionally apply hierarchical parsing
    if use_hierarchical and len(nodes) > 1:
        hierarchical_parser = create_hierarchical_parser()
        # Hierarchical parser works on documents, so we need to reconstruct
        hierarchical_doc = Document(
            text=page_block.content,
            metadata=doc.metadata
        )
        hierarchical_nodes = hierarchical_parser.get_nodes_from_documents([hierarchical_doc])
        
        # Use hierarchical nodes if they provide better structure
        if len(hierarchical_nodes) > 0:
            nodes = hierarchical_nodes
    
    # Post-process nodes to attach page metadata
    for node in nodes:
        # Ensure page metadata is preserved
        if "page_no" not in node.metadata:
            node.metadata["page_no"] = page_block.page_no
        if "page_span" not in node.metadata:
            node.metadata["page_span"] = page_block.page_span
        if "page_nos" not in node.metadata:
            node.metadata["page_nos"] = page_block.page_nos
        if "structure_confidence" not in node.metadata:
            node.metadata["structure_confidence"] = page_block.structure_confidence
        
        # Add repair metadata
        if page_block.repair_applied:
            node.metadata["repair_applied"] = [
                {
                    "repair_type": r.repair_type,
                    "location": r.location,
                    "reason": r.reason
                }
                for r in page_block.repair_applied
            ]
        
        # Preserve node_id for traceability
        if not hasattr(node, "node_id") or node.node_id is None:
            # Generate node_id from content hash
            import hashlib
            node_id = hashlib.sha256(node.text.encode()).hexdigest()[:16]
            node.node_id = node_id
    
    return nodes


def extract_node_metadata(node: BaseNode) -> Dict[str, Any]:
    """Extract metadata from LlamaIndex node.
    
    Args:
        node: LlamaIndex node
    
    Returns:
        Dict[str, Any]: Extracted metadata
    """
    metadata = {
        "node_id": getattr(node, "node_id", None),
        "text": node.text,
        "metadata": node.metadata.copy() if node.metadata else {},
    }
    
    # Extract additional node attributes
    if hasattr(node, "start_char_idx"):
        metadata["start_char_idx"] = node.start_char_idx
    if hasattr(node, "end_char_idx"):
        metadata["end_char_idx"] = node.end_char_idx
    
    return metadata

