"""Magic model post-processing for layout detections."""

import logging
from typing import Dict, List

from src.providers.layout.yolo import LayoutDetectionOutput
from src.utils.bbox import calculate_iou, is_contained

logger = logging.getLogger(__name__)


class MagicModel:
    """Post-processing for layout detections."""

    def __init__(
        self,
        layout_detections: List[LayoutDetectionOutput],
        scale: float = 1.0,
        low_confidence_threshold: float = 0.05,
        high_iou_threshold: float = 0.9,
    ):
        """
        Initialize magic model post-processor.

        Args:
            layout_detections: List of layout detections
            scale: Scale factor for coordinates
            low_confidence_threshold: Threshold for removing low confidence detections
            high_iou_threshold: IoU threshold for deduplication
        """
        self.layout_detections = layout_detections
        self.scale = scale
        self.low_confidence_threshold = low_confidence_threshold
        self.high_iou_threshold = high_iou_threshold

        # Apply fixes
        self._fix_axis()
        self._remove_low_confidence()
        self._remove_high_iou_duplicates()

    def _fix_axis(self):
        """Fix and validate bounding boxes."""
        valid_detections = []
        for det in self.layout_detections:
            bbox = det.bbox
            # Validate bbox coordinates
            if bbox.x0 < bbox.x1 and bbox.y0 < bbox.y1:
                # Scale if needed
                if self.scale != 1.0:
                    bbox.x0 *= self.scale
                    bbox.y0 *= self.scale
                    bbox.x1 *= self.scale
                    bbox.y1 *= self.scale
                valid_detections.append(det)
            else:
                logger.debug(f"Removed invalid bbox: {bbox}")
        self.layout_detections = valid_detections

    def _remove_low_confidence(self):
        """Remove detections with very low confidence."""
        self.layout_detections = [
            det for det in self.layout_detections if det.score >= self.low_confidence_threshold
        ]

    def _remove_high_iou_duplicates(self):
        """Remove overlapping detections, keeping the one with higher confidence."""
        if len(self.layout_detections) <= 1:
            return

        # Sort by confidence (descending)
        sorted_detections = sorted(self.layout_detections, key=lambda x: x.score, reverse=True)

        kept = []
        for det in sorted_detections:
            should_keep = True
            bbox1 = det.bbox.to_tuple()

            for kept_det in kept:
                bbox2 = kept_det.bbox.to_tuple()
                iou = calculate_iou(bbox1, bbox2)

                if iou > self.high_iou_threshold:
                    # High overlap, skip this one (keep the one already in kept)
                    should_keep = False
                    break

            if should_keep:
                kept.append(det)

        self.layout_detections = kept

    def get_detections(self) -> List[LayoutDetectionOutput]:
        """Get processed detections."""
        return self.layout_detections

