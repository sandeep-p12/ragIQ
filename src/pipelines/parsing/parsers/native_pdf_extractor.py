"""Native PDF extraction utilities for creating blocks when layout detection is unavailable."""

import logging
from typing import List, Tuple

import pypdfium2 as pdfium
from pypdfium2._helpers.page import PdfPage

from src.schema.document import BBox, BlockType, ImageBlock, TableBlock, TextBlock, TitleBlock

logger = logging.getLogger(__name__)


def extract_blocks_from_native_pdf(
    pdf_page: PdfPage,
    page_width: int,
    page_height: int,
    page_index: int,
) -> List[Tuple[BlockType, BBox, str]]:
    """
    Extract blocks from native PDF text and images.
    
    Args:
        pdf_page: PDFium page object
        page_width: Page width in points
        page_height: Page height in points
        
    Returns:
        List of (block_type, bbox, text) tuples
    """
    blocks = []
    
    try:
        # Extract text objects
        text_objects = []
        for obj in pdf_page.get_objects():
            if obj.type == 1:  # Text object
                try:
                    pos = obj.get_pos()  # (left, bottom, right, top)
                    text = obj.get_text()
                    if text and text.strip():
                        # Convert to normalized coordinates
                        x0 = pos[0] / page_width if page_width > 0 else 0.0
                        y0 = (page_height - pos[3]) / page_height if page_height > 0 else 0.0
                        x1 = pos[2] / page_width if page_width > 0 else 1.0
                        y1 = (page_height - pos[1]) / page_height if page_height > 0 else 1.0
                        
                        bbox = BBox(x0=x0, y0=y0, x1=x1, y1=y1)
                        text_objects.append((bbox, text.strip()))
                except Exception:
                    continue
        
        # Group text objects into blocks (simple heuristic: group by proximity)
        if text_objects:
            # Sort by Y position (top to bottom)
            text_objects.sort(key=lambda x: x[0].y0)
            
            current_block = []
            current_y = None
            block_y0 = None
            block_y1 = None
            block_x0 = 1.0
            block_x1 = 0.0
            
            for bbox, text in text_objects:
                # Check if this should be part of current block or new block
                if current_y is None or abs(bbox.y0 - current_y) < 0.02:  # Within 2% of page height
                    # Same line or close - add to current block
                    current_block.append((bbox, text))
                    current_y = bbox.y0
                    if block_y0 is None:
                        block_y0 = bbox.y0
                    block_y1 = max(block_y1 or 0, bbox.y1)
                    block_x0 = min(block_x0, bbox.x0)
                    block_x1 = max(block_x1, bbox.x1)
                else:
                    # New block - save previous
                    if current_block:
                        block_text = " ".join(text for _, text in current_block)
                        block_bbox = BBox(x0=block_x0, y0=block_y0, x1=block_x1, y1=block_y1)
                        
                        # Heuristic: determine block type
                        block_type = _classify_text_block(block_text)
                        blocks.append((block_type, block_bbox, block_text))
                    
                    # Start new block
                    current_block = [(bbox, text)]
                    current_y = bbox.y0
                    block_y0 = bbox.y0
                    block_y1 = bbox.y1
                    block_x0 = bbox.x0
                    block_x1 = bbox.x1
            
            # Save last block
            if current_block:
                block_text = " ".join(text for _, text in current_block)
                block_bbox = BBox(x0=block_x0, y0=block_y0, x1=block_x1, y1=block_y1)
                block_type = _classify_text_block(block_text)
                blocks.append((block_type, block_bbox, block_text))
        
        # Extract images from PDF page
        # Note: pypdfium2 doesn't have a direct get_images() method
        # Images embedded in PDF are typically in the content stream
        # For now, we skip direct image extraction and rely on layout detection
        # This is a limitation - in production, you might want to parse the content stream
        # or use a different PDF library for image extraction
        pass
            
    except Exception as e:
        logger.warning(f"Error extracting native PDF blocks: {e}")
    
    return blocks


def _classify_text_block(text: str) -> BlockType:
    """Classify text block type based on heuristics."""
    text_upper = text.upper().strip()
    
    # Check if it's a title/heading (short, all caps, or starts with number)
    if len(text) < 100 and (text_upper == text or text.startswith(("#", "##", "###"))):
        return BlockType.TITLE
    
    # Check if it looks like a table (multiple spaces/tabs, numbers, etc.)
    if "\t" in text or (text.count("  ") > 3 and any(c.isdigit() for c in text)):
        return BlockType.TABLE
    
    return BlockType.TEXT

