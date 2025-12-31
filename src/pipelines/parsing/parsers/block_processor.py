"""Block processing utilities for converting layout/OCR results to blocks."""

import logging
from typing import List, Optional

from src.pipelines.parsing.processing.magic import MagicModel
from src.providers.ocr.doctr import TextDetection
from src.providers.layout.yolo import LayoutDetectionOutput
from src.schema.document import (
    BBox,
    Block,
    BlockType,
    ImageBlock,
    Span,
    TableBlock,
    TextBlock,
    TitleBlock,
)

logger = logging.getLogger(__name__)


def layout_detections_to_blocks(
    layout_detections: List[LayoutDetectionOutput],
    page_width: int,
    page_height: int,
    page_index: int,
) -> List[Block]:
    """
    Convert layout detections to blocks.

    Args:
        layout_detections: List of layout detection outputs
        page_width: Page width in pixels
        page_height: Page height in pixels
        page_index: Page index

    Returns:
        List of blocks
    """
    blocks = []

    for det in layout_detections:
        bbox = det.bbox
        block_type = det.block_type

        if block_type == BlockType.TITLE:
            block = TitleBlock(
                block_type=BlockType.TITLE,
                bbox=bbox,
                page_index=page_index,
                text="",  # Will be filled from OCR/native text
                level=1,
                confidence=det.score,
            )
        elif block_type == BlockType.TEXT:
            block = TextBlock(
                block_type=BlockType.TEXT,
                bbox=bbox,
                page_index=page_index,
                text="",  # Will be filled from OCR/native text
                confidence=det.score,
            )
        elif block_type == BlockType.TABLE:
            block = TableBlock(
                block_type=BlockType.TABLE,
                bbox=bbox,
                page_index=page_index,
                confidence=det.score,
            )
        elif block_type == BlockType.IMAGE:
            block = ImageBlock(
                block_type=BlockType.IMAGE,
                bbox=bbox,
                page_index=page_index,
                confidence=det.score,
            )
        else:
            # Default to text block
            block = TextBlock(
                block_type=BlockType.TEXT,
                bbox=bbox,
                page_index=page_index,
                text="",
                confidence=det.score,
            )

        blocks.append(block)

    return blocks


def fill_text_from_ocr(
    blocks: List[Block],
    ocr_detection: TextDetection,
    page_width: int,
    page_height: int,
) -> List[Block]:
    """
    Fill text blocks with OCR results.

    Args:
        blocks: List of blocks
        ocr_detection: OCR text detection result
        page_width: Page width
        page_height: Page height

    Returns:
        Updated blocks with text filled
    """
    from src.utils.bbox import calculate_overlap_ratio

    # Create spans from OCR
    ocr_spans = []
    if len(ocr_detection.bboxes) != len(ocr_detection.texts) or len(ocr_detection.bboxes) != len(ocr_detection.scores):
        raise ValueError(f"Mismatch: {len(ocr_detection.bboxes)} bboxes, {len(ocr_detection.texts)} texts, {len(ocr_detection.scores)} scores")
    for bbox, text, score in zip(ocr_detection.bboxes, ocr_detection.texts, ocr_detection.scores):
        ocr_spans.append(
            Span(
                text=text,
                bbox=bbox,
                span_type="text",
                confidence=score,
            )
        )

    # Assign spans to blocks and create updated blocks
    updated_blocks = []
    text_filled_count = 0
    
    for block in blocks:
        if isinstance(block, (TextBlock, TitleBlock)) and block.bbox:
            block_bbox = block.bbox.to_tuple()
            block_spans = []

            for span in ocr_spans:
                if span.bbox:
                    span_bbox = span.bbox.to_tuple()
                    overlap = calculate_overlap_ratio(block_bbox, span_bbox)

                    if overlap > 0.3:  # Lower threshold for better matching
                        block_spans.append((overlap, span))

            # Sort spans by overlap and position, then merge text
            if block_spans:
                # Sort by overlap first, then by position
                block_spans.sort(key=lambda x: (-x[0], x[1].bbox.y0 if x[1].bbox else 0, x[1].bbox.x0 if x[1].bbox else 0))
                # Extract just the spans (without overlap scores)
                sorted_spans = [span for _, span in block_spans]
                block_text = " ".join(span.text for span in sorted_spans)
                # Create updated block with text and spans
                updated_block = block.model_copy(update={
                    "text": block_text,
                    "spans": sorted_spans,
                })
                updated_blocks.append(updated_block)
                text_filled_count += 1
            else:
                updated_blocks.append(block)
        else:
            updated_blocks.append(block)
    
    logger.debug(f"Filled text from OCR for {text_filled_count} out of {len([b for b in blocks if isinstance(b, (TextBlock, TitleBlock))])} text blocks")

    return updated_blocks


def fill_text_from_native(
    blocks: List[Block],
    pdf_page,
    page_width: int,
    page_height: int,
) -> List[Block]:
    """
    Fill text blocks from native PDF text extraction.

    Args:
        blocks: List of blocks
        pdf_page: PDFium page object
        page_width: Page width
        page_height: Page height

    Returns:
        Updated blocks with text filled
    """
    from src.utils.bbox import calculate_overlap_ratio

    try:
        textpage = pdf_page.get_textpage()
        
        # Use get_text_bounded() to extract text for each block's bbox
        updated_blocks = []
        text_filled_count = 0
        
        for block in blocks:
            if isinstance(block, (TextBlock, TitleBlock)) and block.bbox:
                try:
                    # Convert normalized bbox to absolute coordinates
                    block_x0, block_y0, block_x1, block_y1 = block.bbox.to_absolute(page_width, page_height)
                    
                    # PDF coordinates use bottom-left origin, but get_text_bounded uses top-left
                    # So we need to invert Y coordinates
                    pdf_y0 = page_height - block_y1  # Convert top to bottom
                    pdf_y1 = page_height - block_y0  # Convert bottom to top
                    
                    # Extract text within this block's bounding box
                    block_text = textpage.get_text_bounded(
                        left=block_x0,
                        top=pdf_y0,  # Use inverted Y
                        right=block_x1,
                        bottom=pdf_y1,  # Use inverted Y
                    )
                    
                    if block_text and block_text.strip():
                        updated_block = block.model_copy(update={"text": block_text.strip()})
                        updated_blocks.append(updated_block)
                        text_filled_count += 1
                    else:
                        # If no text found in bbox, try a slightly larger region (10% padding)
                        padding_x = (block_x1 - block_x0) * 0.1
                        padding_y = (block_y1 - block_y0) * 0.1
                        # Invert Y for padding too
                        padded_pdf_y0 = max(0, page_height - (block_y1 + padding_y))
                        padded_pdf_y1 = min(page_height, page_height - (block_y0 - padding_y))
                        block_text = textpage.get_text_bounded(
                            left=max(0, block_x0 - padding_x),
                            top=padded_pdf_y0,
                            right=min(page_width, block_x1 + padding_x),
                            bottom=padded_pdf_y1,
                        )
                        if block_text and block_text.strip():
                            updated_block = block.model_copy(update={"text": block_text.strip()})
                            updated_blocks.append(updated_block)
                            text_filled_count += 1
                        else:
                            updated_blocks.append(block)
                except Exception as e:
                    logger.warning(f"Error extracting text for block {block.block_id}: {e}")
                    updated_blocks.append(block)
            else:
                updated_blocks.append(block)
        
        logger.debug(f"Filled text for {text_filled_count} out of {len([b for b in blocks if isinstance(b, (TextBlock, TitleBlock))])} text blocks")
        return updated_blocks

    except Exception as e:
        logger.warning(f"Error extracting native text: {e}")
        import traceback
        logger.debug(traceback.format_exc())

    return blocks

