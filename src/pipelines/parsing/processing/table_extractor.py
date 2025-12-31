"""Table extraction and processing."""

import base64
import logging
from io import BytesIO
from typing import List, Optional, Tuple, TYPE_CHECKING

import numpy as np
from bs4 import BeautifulSoup

from src.config.prompts import TABLE_EXTRACTION_PROMPT
from src.schema.document import BBox, Block, BlockType, TableBlock, TextBlock

if TYPE_CHECKING:
    from PIL.Image import Image as PILImage

logger = logging.getLogger(__name__)


def extract_table_text(
    table_bbox: BBox,
    text_blocks: List[TextBlock],
    page_width: int,
    page_height: int,
) -> List[Tuple[BBox, str]]:
    """
    Extract text that intersects with table bbox.
    Uses both block text and spans for more accurate extraction.
    Also attempts to split large text blocks that contain table content.

    Args:
        table_bbox: Table bounding box
        text_blocks: List of text blocks (can include TitleBlock)
        page_width: Page width
        page_height: Page height

    Returns:
        List of (bbox, text) tuples within table region
    """
    from src.utils.bbox import calculate_overlap_ratio
    
    table_x0, table_y0, table_x1, table_y1 = table_bbox.to_absolute(page_width, page_height)
    table_texts = []

    for block in text_blocks:
        if block.bbox is None:
            continue

        block_x0, block_y0, block_x1, block_y1 = block.bbox.to_absolute(page_width, page_height)

        # Check intersection
        if (
            block_x1 < table_x0
            or block_x0 > table_x1
            or block_y1 < table_y0
            or block_y0 > table_y1
        ):
            continue

        # If block has spans, use them for more granular extraction
        if hasattr(block, 'spans') and block.spans:
            for span in block.spans:
                if span.bbox:
                    span_x0, span_y0, span_x1, span_y1 = span.bbox.to_absolute(page_width, page_height)
                    # Check if span overlaps with table
                    span_bbox = (span_x0, span_y0, span_x1, span_y1)
                    table_bbox_tuple = (table_x0, table_y0, table_x1, table_y1)
                    overlap = calculate_overlap_ratio(span_bbox, table_bbox_tuple)
                    if overlap > 0.3:  # Significant overlap
                        table_texts.append((span.bbox, span.text))
        else:
            # Use block text if no spans
            if block.text:
                # Check if this block is mostly within the table (likely table content)
                block_area = (block_x1 - block_x0) * (block_y1 - block_y0)
                table_area = (table_x1 - table_x0) * (table_y1 - table_y0)
                
                # Calculate intersection area
                intersect_x0 = max(block_x0, table_x0)
                intersect_y0 = max(block_y0, table_y0)
                intersect_x1 = min(block_x1, table_x1)
                intersect_y1 = min(block_y1, table_y1)
                
                if intersect_x1 > intersect_x0 and intersect_y1 > intersect_y0:
                    intersect_area = (intersect_x1 - intersect_x0) * (intersect_y1 - intersect_y0)
                    overlap_ratio = intersect_area / block_area if block_area > 0 else 0
                    
                    # If block is mostly within table, try to split it
                    if overlap_ratio > 0.7 and block.text:
                        # Try to split text that might be table cells (separated by spaces/tabs)
                        # This handles cases where native PDF extraction groups table text together
                        text_parts = _split_table_text(block.text, block.bbox, page_width, page_height)
                        if len(text_parts) > 1:
                            # Multiple parts found - use them
                            table_texts.extend(text_parts)
                        else:
                            # Single block - use as is
                            table_texts.append((block.bbox, block.text))
                    else:
                        # Partial overlap - use the block
                        table_texts.append((block.bbox, block.text))

    return table_texts


def _split_table_text(text: str, bbox: BBox, page_width: int, page_height: int) -> List[Tuple[BBox, str]]:
    """
    Attempt to split text that might contain table cells.
    
    Args:
        text: Text to split
        bbox: Bounding box of the text
        page_width: Page width
        page_height: Page height
        
    Returns:
        List of (bbox, text) tuples for potential cells
    """
    # Try splitting by multiple spaces or tabs (common in tables)
    import re
    
    # Split by 2+ spaces or tabs
    parts = re.split(r'\s{2,}|\t+', text.strip())
    parts = [p.strip() for p in parts if p.strip()]
    
    if len(parts) <= 1:
        # Can't split - return original
        return [(bbox, text)]
    
    # Calculate approximate cell width
    x0, y0, x1, y1 = bbox.to_absolute(page_width, page_height)
    block_width = x1 - x0
    cell_width = block_width / len(parts) if len(parts) > 0 else block_width
    
    # Create bboxes for each part
    result = []
    for i, part in enumerate(parts):
        cell_x0 = x0 + (i * cell_width)
        cell_x1 = x0 + ((i + 1) * cell_width)
        cell_bbox = BBox(
            x0=cell_x0 / page_width if page_width > 0 else 0,
            y0=y0 / page_height if page_height > 0 else 0,
            x1=cell_x1 / page_width if page_width > 0 else 1,
            y1=y1 / page_height if page_height > 0 else 1,
        )
        result.append((cell_bbox, part))
    
    return result


def build_table_grid(
    table_texts: List[Tuple[BBox, str]],
    page_width: int,
    page_height: int,
) -> List[List[str]]:
    """
    Build table grid from text positions using clustering.

    Args:
        table_texts: List of (bbox, text) tuples
        page_width: Page width
        page_height: Page height

    Returns:
        2D list representing table cells
    """
    if not table_texts:
        return []

    # Extract coordinates
    y_coords = []
    x_coords = []
    texts = []

    for bbox, text in table_texts:
        x0, y0, x1, y1 = bbox.to_absolute(page_width, page_height)
        y_coords.append((y0 + y1) / 2)  # Center y
        x_coords.append((x0 + x1) / 2)  # Center x
        texts.append(text)

    # Cluster by Y (rows)
    y_array = np.array(y_coords)
    y_sorted_indices = np.argsort(y_array)
    y_sorted = y_array[y_sorted_indices]

    # Group rows (within threshold)
    row_threshold = np.median(np.diff(np.sort(y_array))) * 0.5
    rows = []
    current_row = [y_sorted_indices[0]]
    current_y = y_sorted[0]

    for i in range(1, len(y_sorted)):
        if y_sorted[i] - current_y < row_threshold:
            current_row.append(y_sorted_indices[i])
        else:
            rows.append(current_row)
            current_row = [y_sorted_indices[i]]
            current_y = y_sorted[i]
    rows.append(current_row)

    # Sort each row by X
    table_grid = []
    for row_indices in rows:
        row_x = [x_coords[i] for i in row_indices]
        row_sorted = sorted(zip(row_indices, row_x), key=lambda x: x[1])
        row_texts = [texts[i] for i, _ in row_sorted]
        table_grid.append(row_texts)

    return table_grid


def _extract_table_with_vision_llm(
    table_image: "PILImage",
    table_bbox: BBox,
    config: Optional[object] = None,
) -> Optional[str]:
    """
    Extract table content using vision LLM as final fallback.
    
    Args:
        table_image: Cropped table image
        table_bbox: Table bounding box
        config: ParseForgeConfig for LLM access
        
    Returns:
        Markdown table string, or None if failed
    """
    if config is None:
        return None
    
    # Check if LLM is configured
    if not hasattr(config, 'llm_provider') or config.llm_provider == "none":
        return None
    
    if not hasattr(config, 'llm_api_key') or not config.llm_api_key:
        return None
    
    try:
        from openai import OpenAI
        
        client = OpenAI(api_key=config.llm_api_key)
        
        # Convert table image to base64
        buffered = BytesIO()
        table_image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        
        # Create prompt for table extraction
        prompt = TABLE_EXTRACTION_PROMPT

        # Use vision-capable model
        model = getattr(config, 'llm_model', 'gpt-4o')
        if model not in ["gpt-4o", "gpt-4-vision-preview", "gpt-4-turbo"]:
            if "gpt-4" in model.lower():
                model = "gpt-4o"
            else:
                model = "gpt-4o"
        
        # Call OpenAI vision API
        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{img_str}",
                            },
                        },
                    ],
                }
            ],
            max_tokens=2000,
            temperature=0.1,
        )
        
        markdown_table = response.choices[0].message.content.strip()
        
        # Extract just the table (remove any explanation text)
        import re
        # Find markdown table pattern
        table_pattern = r'(\|.*\|(?:\n\|[:\s\-|]+\|)?(?:\n\|.*\|)*)'
        match = re.search(table_pattern, markdown_table, re.MULTILINE)
        if match:
            markdown_table = match.group(1)
        
        logger.info(f"Extracted table using vision LLM - {len(markdown_table)} chars")
        return markdown_table
        
    except Exception as e:
        logger.warning(f"Vision LLM table extraction failed: {e}")
        import traceback
        logger.debug(f"Vision LLM error traceback: {traceback.format_exc()}")
        return None


def extract_table(
    table_bbox: BBox,
    text_blocks: List[TextBlock],
    page_width: int,
    page_height: int,
    page_image: Optional["PILImage"] = None,
    ocr_engine: Optional[object] = None,
    config: Optional[object] = None,
) -> TableBlock:
    """
    Extract table structure from layout and text.
    Fallback order: text blocks -> OCR -> Vision LLM.

    Args:
        table_bbox: Table bounding box
        text_blocks: List of text blocks
        page_width: Page width
        page_height: Page height
        page_image: Optional page image for OCR/LLM fallback
        ocr_engine: Optional OCR engine (DoctrOCR) for table region OCR
        config: Optional ParseForgeConfig for LLM access

    Returns:
        TableBlock with extracted structure
    """
    # Extract text within table region
    table_texts = extract_table_text(table_bbox, text_blocks, page_width, page_height)
    
    # If no text found, try OCR on table region
    if not table_texts and page_image is not None and ocr_engine is not None:
        logger.info(f"No text found in table bbox, trying OCR on table region")
        try:
            # Crop table region from page image
            table_x0, table_y0, table_x1, table_y1 = table_bbox.to_absolute(page_width, page_height)
            
            # Ensure coordinates are within image bounds
            img_width, img_height = page_image.size
            table_x0 = max(0, int(table_x0))
            table_y0 = max(0, int(table_y0))
            table_x1 = min(img_width, int(table_x1))
            table_y1 = min(img_height, int(table_y1))
            
            if table_x1 > table_x0 and table_y1 > table_y0:
                # Crop table region
                table_crop = page_image.crop((table_x0, table_y0, table_x1, table_y1))
                
                # Run OCR on cropped table
                ocr_results = ocr_engine.ocr([table_crop])
                if ocr_results and len(ocr_results) > 0:
                    ocr_detection = ocr_results[0]
                    
                    # Extract text from OCR results with positions
                    # OCR returns TextDetection with bboxes, texts, scores
                    crop_width = table_x1 - table_x0
                    crop_height = table_y1 - table_y0
                    
                    if len(ocr_detection.bboxes) != len(ocr_detection.texts):
                        raise ValueError(f"Mismatch: {len(ocr_detection.bboxes)} bboxes but {len(ocr_detection.texts)} texts")
                    for bbox, text in zip(ocr_detection.bboxes, ocr_detection.texts):
                        if bbox and text and text.strip():
                            # Convert OCR bbox (relative to crop) to page coordinates
                            # OCR bbox is normalized (0-1) relative to crop
                            crop_x0, crop_y0, crop_x1, crop_y1 = bbox.to_absolute(crop_width, crop_height)
                            
                            # Adjust to page coordinates
                            abs_x0 = (crop_x0 + table_x0) / page_width
                            abs_y0 = (crop_y0 + table_y0) / page_height
                            abs_x1 = (crop_x1 + table_x0) / page_width
                            abs_y1 = (crop_y1 + table_y0) / page_height
                            
                            word_bbox = BBox(x0=abs_x0, y0=abs_y0, x1=abs_x1, y1=abs_y1)
                            table_texts.append((word_bbox, text.strip()))
                    
                    logger.info(f"Extracted {len(table_texts)} text elements from table OCR")
        except Exception as e:
            logger.warning(f"OCR on table region failed: {e}")
            import traceback
            logger.debug(f"OCR error traceback: {traceback.format_exc()}")
    
    # If still no text, try vision LLM as final fallback
    if not table_texts and page_image is not None and config is not None:
        logger.info(f"No text found in table bbox, trying vision LLM on table image")
        try:
            # Crop table region from page image
            table_x0, table_y0, table_x1, table_y1 = table_bbox.to_absolute(page_width, page_height)
            
            # Ensure coordinates are within image bounds
            img_width, img_height = page_image.size
            table_x0 = max(0, int(table_x0))
            table_y0 = max(0, int(table_y0))
            table_x1 = min(img_width, int(table_x1))
            table_y1 = min(img_height, int(table_y1))
            
            if table_x1 > table_x0 and table_y1 > table_y0:
                # Crop table region
                table_crop = page_image.crop((table_x0, table_y0, table_x1, table_y1))
                
                # Extract table using vision LLM
                markdown_table = _extract_table_with_vision_llm(table_crop, table_bbox, config)
                
                if markdown_table:
                    # Parse markdown table to create cells
                    cells = _parse_markdown_table_to_cells(markdown_table)
                    
                    if cells:
                        # Create HTML from cells
                        html = _cells_to_html(cells)
                        
                        # Store markdown in HTML for markdown formatter
                        html_with_markdown = f"<!-- MARKDOWN_TABLE_START -->\n{markdown_table}\n<!-- MARKDOWN_TABLE_END -->\n{html}"
                        
                        logger.info(f"Extracted table using vision LLM - {len(cells)} rows, {max(len(row) for row in cells) if cells else 0} cols")
                        
                        return TableBlock(
                            block_type=BlockType.TABLE,
                            bbox=table_bbox,
                            page_index=0,  # Will be set by caller
                            html=html_with_markdown,
                            cells=cells,
                            num_rows=len(cells),
                            num_cols=max(len(row) for row in cells) if cells else 0,
                        )
        except Exception as e:
            logger.warning(f"Vision LLM table extraction failed: {e}")
            import traceback
            logger.debug(f"Vision LLM error traceback: {traceback.format_exc()}")
    
    if not table_texts:
        logger.warning(f"No text found within table bbox {table_bbox} (tried text blocks, OCR, and vision LLM)")
        # Return empty table
        return TableBlock(
            block_type=BlockType.TABLE,
            bbox=table_bbox,
            page_index=0,
            html="<table></table>",
            cells=[],
            num_rows=0,
            num_cols=0,
        )
    
    logger.debug(f"Extracted {len(table_texts)} text elements for table")
    if len(table_texts) <= 5:  # Log first few for debugging
        logger.debug(f"Table texts: {[(t[1][:50], t[0]) for t in table_texts[:5]]}")

    # Build grid
    cells = build_table_grid(table_texts, page_width, page_height)
    
    if not cells:
        logger.warning(f"Failed to build table grid from {len(table_texts)} text elements")
    else:
        logger.debug(f"Built table grid: {len(cells)} rows, {max(len(row) for row in cells) if cells else 0} cols")

    # Create HTML table
    html = _cells_to_html(cells)

    return TableBlock(
        block_type=BlockType.TABLE,
        bbox=table_bbox,
        page_index=0,  # Will be set by caller
        html=html,
        cells=cells,
        num_rows=len(cells),
        num_cols=max(len(row) for row in cells) if cells else 0,
    )


def _parse_markdown_table_to_cells(markdown_table: str) -> List[List[str]]:
    """
    Parse markdown table string into 2D cell list.
    
    Args:
        markdown_table: Markdown table string
        
    Returns:
        2D list of cells
    """
    cells = []
    lines = markdown_table.strip().split('\n')
    
    for line in lines:
        line = line.strip()
        if not line or not line.startswith('|'):
            continue
        
        # Skip separator rows (| --- | --- |)
        if all(c in ['|', '-', ':', ' '] for c in line.replace('|', '')):
            continue
        
        # Extract cells (split by |, remove empty first/last)
        cell_values = [cell.strip() for cell in line.split('|')[1:-1]]
        if cell_values:
            cells.append(cell_values)
    
    return cells


def _cells_to_html(cells: List[List[str]]) -> str:
    """Convert cell grid to HTML table."""
    if not cells:
        return "<table></table>"

    soup = BeautifulSoup("", "html.parser")
    table = soup.new_tag("table")

    for row in cells:
        tr = soup.new_tag("tr")
        for cell_text in row:
            td = soup.new_tag("td")
            td.string = cell_text
            tr.append(td)
        table.append(tr)

    return str(table)

