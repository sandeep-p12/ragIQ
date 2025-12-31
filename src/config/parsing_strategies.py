"""Strategy selection logic for ParseForge."""

import logging
from enum import Enum
from typing import TYPE_CHECKING, List, Optional

import numpy as np
import pypdfium2 as pdfium
from pypdfium2._helpers.page import PdfPage

if TYPE_CHECKING:
    from src.providers.ocr.doctr import TextDetection
else:
    # Forward reference for runtime
    TextDetection = None

logger = logging.getLogger(__name__)


class StrategyEnum(str, Enum):
    """Parsing strategy options."""

    FAST = "fast"  # Native text extraction
    HI_RES = "hi_res"  # OCR-based
    AUTO = "auto"  # Automatic selection
    LLM_FULL = "llm_full"  # Full document parsing using LLM vision


class PageStrategy:
    """Strategy information for a page."""

    def __init__(self, page_index: int, strategy: StrategyEnum, iou: Optional[float] = None):
        self.page_index = page_index
        self.strategy = strategy
        self.iou = iou


def get_page_strategy(
    pdfium_page: PdfPage,
    doctr_detection: Optional["TextDetection"],
    threshold: float = 0.6,
) -> StrategyEnum:
    """
    Determine strategy for a single page using IoU comparison.

    Args:
        pdfium_page: PDFium page object
        doctr_detection: Doctr text detection result
        threshold: IoU threshold (if IoU < threshold, use HI_RES)

    Returns:
        StrategyEnum: FAST or HI_RES
    """
    if doctr_detection is None:
        return StrategyEnum.FAST

    p_width, p_height = int(pdfium_page.get_width()), int(pdfium_page.get_height())

    # Create canvas for PDFium text regions
    pdfium_canvas = np.zeros((p_height, p_width), dtype=np.uint8)

    # Extract text objects from PDFium
    text_coords = []
    for obj in pdfium_page.get_objects():
        if obj.type == 1:  # Text object
            try:
                pos = obj.get_pos()
                text_coords.append(pos)
            except Exception:
                continue

    # Fill PDFium canvas
    for coords in text_coords:
        # PDFium coordinates: (left, bottom, right, top)
        # Convert to image coordinates: (x0, y0, x1, y1)
        x0 = int(p_height - coords[3])
        y0 = int(coords[0])
        x1 = int(p_height - coords[1])
        y1 = int(coords[2])

        x0 = max(0, min(p_height, x0))
        y0 = max(0, min(p_width, y0))
        x1 = max(0, min(p_height, x1))
        y1 = max(0, min(p_width, y1))

        if x1 > x0 and y1 > y0:
            pdfium_canvas[x0:x1, y0:y1] = 1

    # Create canvas for Doctr detections
    doctr_canvas = np.zeros((p_height, p_width), dtype=np.uint8)

    for bbox in doctr_detection.bboxes:
        # Convert normalized bbox to absolute coordinates
        x0, y0, x1, y1 = bbox.to_absolute(p_width, p_height)

        x0 = max(0, min(p_width, x0))
        y0 = max(0, min(p_height, y0))
        x1 = max(0, min(p_width, x1))
        y1 = max(0, min(p_height, y1))

        if x1 > x0 and y1 > y0:
            # Note: bbox uses (x, y) but canvas uses (row, col) = (y, x)
            doctr_canvas[y0:y1, x0:x1] = 1

    # Calculate IoU
    intersection = np.logical_and(pdfium_canvas, doctr_canvas)
    union = np.logical_or(pdfium_canvas, doctr_canvas)

    sum_intersection = np.sum(intersection)
    sum_union = np.sum(union)

    iou = sum_intersection / sum_union if sum_union > 0 else 0.0

    logger.debug(f"Page IoU: {iou:.3f}, threshold: {threshold}")

    if iou < threshold:
        return StrategyEnum.HI_RES
    return StrategyEnum.FAST


def determine_global_strategy(
    page_strategies: List[PageStrategy],
    document_threshold: float = 0.2,
) -> StrategyEnum:
    """
    Determine document-level strategy based on page strategies.

    Args:
        page_strategies: List of page strategies
        document_threshold: Threshold for document-level decision
                           (if >X% pages need HI_RES, use HI_RES for all)

    Returns:
        StrategyEnum: FAST or HI_RES
    """
    if not page_strategies:
        return StrategyEnum.FAST

    hi_res_count = sum(1 for ps in page_strategies if ps.strategy == StrategyEnum.HI_RES)
    hi_res_ratio = hi_res_count / len(page_strategies)

    logger.info(f"Document strategy: {hi_res_ratio:.2%} pages need HI_RES")

    if hi_res_ratio > document_threshold:
        return StrategyEnum.HI_RES
    return StrategyEnum.FAST

