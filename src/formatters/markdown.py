"""Markdown formatter for ParseForge."""

import logging
import re
from typing import Dict, List, Optional

from src.formatters.image import ImageVisionLLMFormatter
from src.schema.document import Block, BlockType, CodeBlock, ImageBlock, ListBlock, TableBlock, TextBlock, TitleBlock

logger = logging.getLogger(__name__)


def escape_markdown(text: str) -> str:
    """Escape special markdown characters."""
    # Characters that need escaping in markdown
    special_chars = r"\_*[]()~`>#+-=|{}.!"
    for char in special_chars:
        text = text.replace(char, f"\\{char}")
    return text


def sanitize_table_cell(cell: str) -> str:
    """
    Sanitize table cell content for markdown.
    
    - Replace newlines with spaces
    - Escape pipe characters (or replace with alternative)
    - Strip and normalize whitespace
    
    Args:
        cell: Raw cell content
        
    Returns:
        Sanitized cell content safe for markdown tables
    """
    if not cell:
        return ""
    
    # Convert to string and strip
    cell_str = str(cell).strip()
    
    # Replace newlines and multiple spaces with single space
    cell_str = re.sub(r'\s+', ' ', cell_str)
    
    # Replace pipe characters with a visually similar alternative or escape
    # Using HTML entity or similar character, but markdown doesn't support entities well
    # So we'll use a different approach: replace | with │ (box drawing character)
    # Or better: just escape it, but markdown tables don't support escaping in cells
    # Best: replace with a space or remove, or use HTML entity
    # Actually, for markdown compatibility, we'll replace | with a similar character
    # Using U+007C (pipe) replacement: we can't escape it in markdown table cells
    # So we'll replace it with a visually similar character: │ (U+2502) or just remove it
    # For now, let's replace with a space to avoid breaking table structure
    cell_str = cell_str.replace('|', ' ')
    
    return cell_str


def format_title_block(block: TitleBlock) -> str:
    """Format title block to markdown."""
    level = block.level
    prefix = "#" * level
    return f"{prefix} {block.text}\n\n"


def format_text_block(block: TextBlock) -> str:
    """Format text block to markdown with English-only formatting."""
    text = block.text.strip()
    if not text:
        return ""

    # Normalize whitespace (standard English spacing)
    text = re.sub(r"\s+", " ", text)

    return f"{text}\n\n"


def format_list_block(block: ListBlock) -> str:
    """Format list block to markdown."""
    if not block.items:
        return ""

    result = []
    marker = "1. " if block.ordered else "- "

    for item in block.items:
        result.append(f"{marker}{item}")

    return "\n".join(result) + "\n\n"


def format_table_block(block: TableBlock) -> str:
    """Format table block to markdown with proper cell sanitization."""
    # Check if LLM-formatted markdown is stored in HTML
    if block.html and "<!-- MARKDOWN_TABLE_START -->" in block.html:
        # Extract LLM-formatted markdown
        start_marker = "<!-- MARKDOWN_TABLE_START -->\n"
        end_marker = "\n<!-- MARKDOWN_TABLE_END -->"
        start_idx = block.html.find(start_marker)
        if start_idx != -1:
            start_idx += len(start_marker)
            end_idx = block.html.find(end_marker, start_idx)
            if end_idx != -1:
                markdown_table = block.html[start_idx:end_idx].strip()
                if markdown_table:
                    # Clean up the extracted markdown table
                    markdown_table = _clean_markdown_table(markdown_table)
                    return markdown_table + "\n\n"
    
    # Fallback to HTML conversion
    if block.html:
        # Try to extract from HTML
        from bs4 import BeautifulSoup

        soup = BeautifulSoup(block.html, "html.parser")
        rows = soup.find_all("tr")

        if not rows:
            return ""

        # Extract header
        header_row = rows[0]
        headers = [sanitize_table_cell(cell.get_text()) for cell in header_row.find_all(["th", "td"])]

        # Extract data rows
        data_rows = []
        for row in rows[1:]:
            cells = [sanitize_table_cell(cell.get_text()) for cell in row.find_all(["td", "th"])]
            data_rows.append(cells)

        # Ensure all rows have same number of columns as header
        num_cols = len(headers)
        for row in data_rows:
            while len(row) < num_cols:
                row.append("")
            row[:] = row[:num_cols]  # Truncate if too many

        # Build markdown table
        result = []
        result.append("| " + " | ".join(headers) + " |")
        result.append("| " + " | ".join(["---"] * num_cols) + " |")

        for row in data_rows:
            result.append("| " + " | ".join(row) + " |")

        return "\n".join(result) + "\n\n"

    elif block.cells:
        # Use cell data directly
        if not block.cells:
            return ""

        # First row as header
        headers = [sanitize_table_cell(str(h)) for h in (block.cells[0] if block.cells else [])]
        num_cols = len(headers)
        
        result = []
        result.append("| " + " | ".join(headers) + " |")
        result.append("| " + " | ".join(["---"] * num_cols) + " |")

        for row in block.cells[1:]:
            # Sanitize and pad/truncate to match header columns
            cells = [sanitize_table_cell(str(cell)) for cell in row]
            while len(cells) < num_cols:
                cells.append("")
            cells = cells[:num_cols]
            result.append("| " + " | ".join(cells) + " |")

        return "\n".join(result) + "\n\n"

    return ""


def _clean_markdown_table(table_markdown: str) -> str:
    """
    Clean and validate a markdown table string.
    
    Ensures:
    - All rows have consistent column counts
    - No broken pipe characters
    - Proper separator row
    - No empty rows
    
    Args:
        table_markdown: Raw markdown table string
        
    Returns:
        Cleaned markdown table string
    """
    lines = [line.rstrip() for line in table_markdown.split('\n')]
    table_lines = []
    max_cols = 0
    
    # First pass: find max columns and identify separator
    for line in lines:
        line = line.strip()
        if not line:
            continue
        if line.startswith('|') and line.endswith('|'):
            # Count columns
            cells = [c.strip() for c in line.split('|')[1:-1]]
            num_cols = len(cells)
            max_cols = max(max_cols, num_cols)
            
            # Check if separator row
            is_separator = all(c in ['-', ':', ' '] or not c for c in cells)
            if not is_separator:
                table_lines.append((cells, False))
            else:
                table_lines.append((None, True))  # Separator will be regenerated
    
    if max_cols == 0:
        return table_markdown  # Return original if no valid table found
    
    # Second pass: rebuild table with consistent columns
    result = []
    separator_added = False
    
    for cells, is_separator in table_lines:
        if is_separator:
            if not separator_added:
                result.append("| " + " | ".join(["---"] * max_cols) + " |")
                separator_added = True
        else:
            # Pad or truncate cells
            while len(cells) < max_cols:
                cells.append("")
            cells = cells[:max_cols]
            # Sanitize each cell
            sanitized_cells = [sanitize_table_cell(cell) for cell in cells]
            result.append("| " + " | ".join(sanitized_cells) + " |")
    
    return "\n".join(result)


def format_image_block(
    block: ImageBlock,
    image_formatter: Optional[ImageVisionLLMFormatter] = None,
    image_descriptions: Optional[Dict[str, str]] = None,
) -> str:
    """
    Format image block to markdown with optional description.

    Args:
        block: Image block to format
        image_formatter: Optional vision LLM formatter for generating descriptions
        image_descriptions: Optional pre-computed descriptions dict (block_id -> description)

    Returns:
        Markdown string with image and description
    """
    # Only format if we have actual image data or path
    # Skip images without data - likely false positives from layout detection
    if not block.image_data and not block.image_path:
        logger.debug(f"Skipping image block {block.block_id} - no image data or path")
        return ""
    
    alt_text = block.alt_text or block.caption or "Image"
    image_path = block.image_path or ""

    # Get description
    description = None
    if image_descriptions and block.block_id in image_descriptions:
        description = image_descriptions[block.block_id]
    elif image_formatter:
        try:
            description = image_formatter.describe_image(block)
        except Exception as e:
            logger.warning(f"Failed to generate image description: {e}")

    # Format with description tag if available
    result = []
    if description:
        description_tag = f"[IMAGE_DESCRIPTION: {description}]"
        result.append(description_tag)
        result.append("")

    # Image markdown
    image_md = f"![{alt_text}]({image_path})"
    result.append(image_md)

    # Caption if available
    if block.caption:
        result.append("")
        result.append(f"*{block.caption}*")

    result.append("")
    result.append("")

    return "\n".join(result)


def format_code_block(block: CodeBlock) -> str:
    """Format code block to markdown."""
    language = block.language or ""
    return f"```{language}\n{block.code}\n```\n\n"


def blocks_to_markdown(
    blocks: List[Block],
    image_formatter: Optional[ImageVisionLLMFormatter] = None,
    image_descriptions: Optional[Dict[str, str]] = None,
) -> str:
    """
    Convert list of blocks to markdown.

    Args:
        blocks: List of blocks to convert
        image_formatter: Optional vision LLM formatter for generating image descriptions
        image_descriptions: Optional pre-computed image descriptions dict (block_id -> description)

    Returns:
        Markdown string
    """
    result = []

    for block in blocks:
        if isinstance(block, TitleBlock):
            result.append(format_title_block(block))
        elif isinstance(block, TextBlock):
            result.append(format_text_block(block))
        elif isinstance(block, ListBlock):
            result.append(format_list_block(block))
        elif isinstance(block, TableBlock):
            result.append(format_table_block(block))
        elif isinstance(block, ImageBlock):
            result.append(format_image_block(block, image_formatter, image_descriptions))
        elif isinstance(block, CodeBlock):
            result.append(format_code_block(block))
        else:
            # Fallback for unknown block types
            if hasattr(block, "text"):
                result.append(f"{block.text}\n\n")

    return "".join(result)

