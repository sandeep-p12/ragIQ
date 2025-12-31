"""Extract typed elements from LlamaIndex nodes using AST parsing (markdown-it-py) instead of regex.

This hybrid approach uses:
- markdown-it-py for accurate AST-based element extraction (reduces regex dependency)
- LlamaIndex for hierarchical node parsing
- Regex only for custom blocks ([IMAGE], [HEADER]) and page markers
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

from llama_index.core.schema import BaseNode
from markdown_it import MarkdownIt

# Import element types from schema
from src.schema.chunk import (
    CustomHeader,
    Element,
    Heading,
    ImageBlockElement,
    ListElement,
    Paragraph,
    Table,
)

# Re-export for backward compatibility
ImageBlock = ImageBlockElement  # Alias for backward compatibility
List = ListElement  # Alias for backward compatibility


# Initialize markdown-it parser once (reusable)
_md_parser = MarkdownIt("commonmark").enable(["table", "strikethrough"])


def _extract_text_from_tokens(tokens: List) -> str:
    """Extract plain text from markdown-it tokens."""
    text_parts = []
    for token in tokens:
        if token.type == "text":
            text_parts.append(token.content)
        elif token.type == "inline" and token.children:
            # Recursively extract from inline tokens
            text_parts.append(_extract_text_from_tokens(token.children))
        elif hasattr(token, "content") and token.content and token.type != "inline":
            text_parts.append(token.content)
    return " ".join(text_parts).strip()


def _extract_list_items_from_tokens(tokens: List, start_idx: int) -> Tuple[List[str], int]:
    """Extract list items from tokens starting at start_idx.
    
    Returns:
        Tuple[List[str], int]: (list of items, index after list)
    """
    items = []
    i = start_idx
    
    # Skip list_open token
    if i < len(tokens) and tokens[i].type in ("bullet_list_open", "ordered_list_open"):
        i += 1
    
    # Extract items until list_close
    while i < len(tokens):
        token = tokens[i]
        
        if token.type in ("bullet_list_close", "ordered_list_close"):
            i += 1
            break
        
        if token.type == "list_item_open":
            # Extract content until list_item_close
            item_tokens = []
            i += 1
            depth = 1
            
            while i < len(tokens) and depth > 0:
                t = tokens[i]
                if t.type == "list_item_open":
                    depth += 1
                elif t.type == "list_item_close":
                    depth -= 1
                    if depth == 0:
                        i += 1
                        break
                
                if depth > 0:
                    item_tokens.append(t)
                i += 1
            
            # Extract text from item tokens
            item_text = _extract_text_from_tokens(item_tokens)
            if item_text:
                items.append(item_text)
        else:
            i += 1
    
    return items, i


def _extract_table_from_tokens(tokens: List, start_idx: int) -> Tuple[Optional[str], List[str], int]:
    """Extract table structure from tokens starting at start_idx.
    
    Returns:
        Tuple[Optional[str], List[str], int]: (header_row, rows, index after table)
    """
    header_row = None
    rows = []
    i = start_idx
    
    # Skip table_open
    if i < len(tokens) and tokens[i].type == "table_open":
        i += 1
    
    in_header = False
    current_row = []
    
    while i < len(tokens):
        token = tokens[i]
        
        if token.type == "table_close":
            i += 1
            break
        
        if token.type == "thead_open":
            in_header = True
            i += 1
        elif token.type == "thead_close":
            in_header = False
            i += 1
        elif token.type == "tbody_open":
            in_header = False
            i += 1
        elif token.type == "tr_open":
            current_row = []
            i += 1
        elif token.type == "tr_close":
            if current_row:
                row_text = " | ".join(current_row)
                if in_header and header_row is None:
                    header_row = row_text
                else:
                    rows.append(row_text)
                current_row = []
            i += 1
        elif token.type in ("th_open", "td_open"):
            # Extract cell content
            cell_tokens = []
            i += 1
            depth = 1
            
            while i < len(tokens) and depth > 0:
                t = tokens[i]
                if t.type in ("th_open", "td_open"):
                    depth += 1
                elif t.type in ("th_close", "td_close"):
                    depth -= 1
                    if depth == 0:
                        i += 1
                        break
                
                if depth > 0:
                    cell_tokens.append(t)
                i += 1
            
            cell_text = _extract_text_from_tokens(cell_tokens)
            current_row.append(cell_text)
        else:
            i += 1
    
    return header_row, rows, i


def _extract_indentation_from_text(text: str, line_start: int) -> str:
    """Extract indentation from the first line of text."""
    lines = text.split("\n")
    if lines:
        first_line = lines[0]
        indent_match = re.match(r'^(\s*)', first_line)
        return indent_match.group(1) if indent_match else ""
    return ""


def extract_elements_from_nodes(nodes: List[BaseNode]) -> List[Element]:
    """Extract typed elements from LlamaIndex nodes using AST parsing.
    
    Uses markdown-it-py for accurate AST-based parsing instead of regex.
    This significantly reduces regex dependency and improves accuracy for
    complex/inconsistent markdown.
    
    Args:
        nodes: List of LlamaIndex nodes
    
    Returns:
        List[Element]: List of extracted elements
    """
    elements = []
    
    for node in nodes:
        node_id = getattr(node, "node_id", None) or f"node_{hash(node.text) % 10000}"
        text = node.text
        lines = text.split("\n")
        
        # Determine line range (approximate)
        start_char = getattr(node, "start_char_idx", 0)
        end_char = getattr(node, "end_char_idx", len(text))
        
        # Approximate line numbers from character positions
        line_start = text[:start_char].count("\n") if start_char > 0 else 0
        line_end = text[:end_char].count("\n") if end_char > 0 else len(lines) - 1
        
        # Check for custom blocks first (still use regex for these custom formats)
        # [IMAGE] and [HEADER] are custom formats, not standard markdown
        if "[IMAGE]" in text and "[/IMAGE]" in text:
            image_match = re.search(r'\[IMAGE\](.*?)\[/IMAGE\]', text, re.DOTALL)
            if image_match:
                extracted_text = image_match.group(1).strip()
                elements.append(ImageBlock(
                    raw_text=text,
                    extracted_text=extracted_text,
                    node_id=node_id,
                    line_start=line_start,
                    line_end=line_end
                ))
                continue
        
        if "[HEADER]" in text and "[/HEADER]" in text:
            header_match = re.search(r'\[HEADER\](.*?)\[/HEADER\]', text, re.DOTALL)
            if header_match:
                header_text = header_match.group(1).strip()
                elements.append(CustomHeader(
                    text=header_text,
                    node_id=node_id,
                    line_start=line_start,
                    line_end=line_end
                ))
                continue
        
        # Parse markdown using AST (markdown-it-py)
        # This replaces all regex-based element detection
        try:
            tokens = _md_parser.parse(text)
            
            # Process tokens to extract elements
            # A single node can contain multiple elements (e.g., heading + paragraph)
            node_elements = _extract_elements_from_tokens(
                tokens, node_id, line_start, line_end, text
            )
            
            if node_elements:
                elements.extend(node_elements)
            else:
                # If no elements extracted, treat as paragraph (fallback)
                if text.strip():
                    elements.append(Paragraph(
                        text=text.strip(),
                        node_id=node_id,
                        line_start=line_start,
                        line_end=line_end
                    ))
            
        except Exception as e:
            # Fallback: if AST parsing fails, treat as paragraph
            # This handles edge cases and malformed markdown gracefully
            if text.strip():
                elements.append(Paragraph(
                    text=text.strip(),
                    node_id=node_id,
                    line_start=line_start,
                    line_end=line_end
                ))
    
    return elements


def _extract_elements_from_tokens(
    tokens: List,
    node_id: str,
    line_start: int,
    line_end: int,
    original_text: str
) -> List[Element]:
    """Extract elements from markdown-it tokens.
    
    This is the core AST-based extraction that replaces regex.
    Works with flat token structure from markdown-it-py.
    """
    elements = []
    i = 0
    
    while i < len(tokens):
        token = tokens[i]
        element = None
        
        if token.type == "heading_open":
            # Extract heading level and text
            level = int(token.tag[1])  # h1, h2, etc. -> 1, 2, etc.
            
            # Find heading_close and extract text between
            heading_tokens = []
            i += 1
            while i < len(tokens):
                t = tokens[i]
                if t.type == "heading_close":
                    i += 1
                    break
                heading_tokens.append(t)
                i += 1
            
            heading_text = _extract_text_from_tokens(heading_tokens)
            
            if heading_text:
                element = Heading(
                    level=level,
                    text=heading_text,
                    node_id=node_id,
                    line_start=line_start,
                    line_end=line_end
                )
        
        elif token.type == "table_open":
            # Extract table structure
            header_row, rows, i = _extract_table_from_tokens(tokens, i)
            
            # Create table signature
            num_cols = len(header_row.split("|")) - 1 if header_row else 0
            num_rows = len(rows)
            signature = f"table_{num_cols}x{num_rows}"
            
            # Check if table candidate (malformed)
            is_candidate = num_cols == 0 or num_rows == 0
            
            element = Table(
                raw_md=original_text,
                header_row=header_row,
                rows=rows,
                signature=signature,
                node_id=node_id,
                line_start=line_start,
                line_end=line_end,
                is_table_candidate=is_candidate
            )
        
        elif token.type in ("bullet_list_open", "ordered_list_open"):
            # Extract list items
            ordered = token.type == "ordered_list_open"
            items, i = _extract_list_items_from_tokens(tokens, i)
            
            if items:
                # Extract indentation from original text
                nesting = _extract_indentation_from_text(original_text, line_start)
                
                element = List(
                    ordered=ordered,
                    items=items,
                    nesting=nesting,
                    node_id=node_id,
                    line_start=line_start,
                    line_end=line_end
                )
        
        elif token.type == "paragraph_open":
            # Extract paragraph text
            para_tokens = []
            i += 1
            while i < len(tokens):
                t = tokens[i]
                if t.type == "paragraph_close":
                    i += 1
                    break
                para_tokens.append(t)
                i += 1
            
            para_text = _extract_text_from_tokens(para_tokens)
            
            if para_text:
                element = Paragraph(
                    text=para_text,
                    node_id=node_id,
                    line_start=line_start,
                    line_end=line_end
                )
        else:
            i += 1
        
        # Add element if found
        if element:
            elements.append(element)
    
    return elements


def generate_section_label(
    element: Element,
    header_path: Optional[str],
    page_no: int,
    block_idx: int
) -> str:
    """Generate section label for element using AST-extracted information.
    
    Priority order:
    1. Element is a heading/custom header (use its text directly)
    2. header_path (from nearest preceding heading)
    3. Meaningful text from element (first sentence, list items, table header)
    4. Fallback to descriptive label
    
    Args:
        element: Element to generate label for
        header_path: Optional header path from hierarchy
        page_no: Page number
        block_idx: Block index within page
    
    Returns:
        str: Section label
    """
    # Priority 1: Element itself is a heading or custom header
    if isinstance(element, Heading):
        return element.text.strip()
    if isinstance(element, CustomHeader):
        return element.text.strip()
    
    # Priority 2: Use header_path if available
    if header_path:
        return header_path
    
    # Priority 3: Extract meaningful text from element
    text = ""
    if isinstance(element, Paragraph):
        # Use first sentence or first 50 chars
        text = element.text
        # Try to get first sentence
        sentences = text.split('.')
        if sentences and len(sentences[0]) > 10:
            text = sentences[0].strip()
        else:
            text = text[:50].strip()
    elif isinstance(element, List):
        # Use first few list items
        if element.items:
            text = " | ".join(element.items[:3])
        else:
            text = "List"
    elif isinstance(element, Table):
        # Use table header
        text = element.header_row or "Table"
    elif isinstance(element, ImageBlock):
        # Use first part of extracted text
        text = element.extracted_text[:50] if element.extracted_text else "Image"
    
    # Clean up text for label
    if text:
        # Remove markdown formatting
        text = re.sub(r'\*\*([^*]+)\*\*', r'\1', text)  # Bold
        text = re.sub(r'\*([^*]+)\*', r'\1', text)  # Italic
        text = re.sub(r'`([^`]+)`', r'\1', text)  # Code
        text = text.strip()
        
        # Use first 60 chars max
        if len(text) > 60:
            text = text[:57] + "..."
        
        if text:
            return text
    
    # Priority 4: Fallback to descriptive label
    element_type = type(element).__name__.lower()
    return f"{element_type.title()} (page {page_no})"


def build_header_path_stack(elements: List[Element]) -> Dict[int, str]:
    """Build header path stack from elements.
    
    Tracks H1→H2→H3 hierarchy when available.
    Elements without headings inherit the nearest preceding heading.
    
    Args:
        elements: List of elements
    
    Returns:
        Dict[int, str]: Mapping of element index to header path
    """
    header_stack = []
    header_paths = {}
    
    for i, element in enumerate(elements):
        # Update stack based on heading level
        if isinstance(element, Heading):
            level = element.level
            
            # Remove headings at same or deeper level
            header_stack = [h for h in header_stack if h[0] < level]
            
            # Add current heading
            header_stack.append((level, element.text.strip()))
        
        elif isinstance(element, CustomHeader):
            # Custom headers are treated as level 1
            header_stack = [(1, element.text.strip())]
        
        # Build path string - always use header_stack if available
        if header_stack:
            path_parts = [h[1] for h in header_stack]
            header_paths[i] = " > ".join(path_parts)
    
    return header_paths
