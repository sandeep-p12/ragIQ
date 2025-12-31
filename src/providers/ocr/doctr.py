"""Doctr OCR implementation for ParseForge."""

import logging
from typing import List, Optional, Tuple

import numpy as np
import onnxruntime as rt
from onnxtr.io import Document
from onnxtr.models import detection_predictor, recognition_predictor
from onnxtr.models.engine import EngineConfig
from PIL import Image
from PIL.Image import Image as PILImage

from src.config.parsing import ParseForgeConfig
from src.schema.document import BBox, Span
from src.utils.exceptions import OCRError

logger = logging.getLogger(__name__)


class TextDetection:
    """Text detection result for a page."""

    def __init__(
        self,
        bboxes: List[BBox],
        texts: List[str],
        scores: List[float],
        page_index: int,
        dimensions: Tuple[int, int],
    ):
        self.bboxes = bboxes
        self.texts = texts
        self.scores = scores
        self.page_index = page_index
        self.dimensions = dimensions  # (width, height)


class DoctrOCR:
    """Doctr OCR implementation using onnxtr."""

    def __init__(
        self,
        device: str = "cpu",
        det_arch: str = "fast_base",
        reco_arch: str = "crnn_vgg16_bn",
        batch_size: int = 32,
        config: Optional[ParseForgeConfig] = None,
    ):
        """
        Initialize Doctr OCR.

        Args:
            device: Device to use (cpu, cuda, mps, coreml)
            det_arch: Detection architecture
            reco_arch: Recognition architecture
            batch_size: Batch size for processing
            config: ParseForge configuration
        """
        self.config = config or ParseForgeConfig()
        self.device = device
        self.batch_size = batch_size

        # Setup ONNX runtime
        session_options = rt.SessionOptions()
        providers = self._get_providers(device)
        engine_config = EngineConfig(
            session_options=session_options,
            providers=providers,
        )

        # Initialize predictors
        try:
            self.det_predictor = detection_predictor(
                arch=det_arch,
                assume_straight_pages=True,
                preserve_aspect_ratio=True,
                symmetric_pad=True,
                batch_size=batch_size,
                load_in_8_bit=False,
                engine_cfg=engine_config,
            )

            self.reco_predictor = recognition_predictor(
                arch=reco_arch,
                batch_size=batch_size,
                load_in_8_bit=False,
                engine_cfg=engine_config,
            )

            logger.info(f"Initialized Doctr OCR with det={det_arch}, reco={reco_arch}")
        except Exception as e:
            logger.warning(f"Failed to initialize Doctr OCR: {e}. OCR will be disabled.")
            self.det_predictor = None
            self.reco_predictor = None

    def _get_providers(self, device: str) -> List[str]:
        """Get ONNX runtime providers based on device."""
        if device == "cuda":
            return ["CUDAExecutionProvider", "CPUExecutionProvider"]
        elif device == "mps":
            return ["CoreMLExecutionProvider", "CPUExecutionProvider"]
        elif device == "coreml":
            return ["CoreMLExecutionProvider", "CPUExecutionProvider"]
        else:
            return ["CPUExecutionProvider"]

    def detect_text(self, pages: List[PILImage]) -> List[TextDetection]:
        """
        Detect text regions in pages (detection only).

        Args:
            pages: List of PIL images

        Returns:
            List of TextDetection objects
        """
        if self.det_predictor is None:
            # Return empty detections if OCR not initialized
            return [
                TextDetection(
                    bboxes=[],
                    texts=[],
                    scores=[],
                    page_index=i,
                    dimensions=(page.width, page.height),
                )
                for i, page in enumerate(pages)
            ]
        
        rasterized_pages = [np.array(page) for page in pages]

        if any(page.ndim != 3 for page in rasterized_pages):
            raise OCRError("All pages must be multi-channel 2D images")

        try:
            # Run detection
            loc_preds, _ = self.det_predictor(rasterized_pages, return_maps=True)

            results = []
            if len(rasterized_pages) != len(loc_preds):
                raise ValueError(f"Mismatch: {len(rasterized_pages)} pages but {len(loc_preds)} detections")
            for page_idx, (page, loc_pred) in enumerate(zip(rasterized_pages, loc_preds)):
                img_h, img_w = page.shape[:2]
                bboxes = []
                scores = []

                # Convert detections to BBox format
                for detection in loc_pred:
                    # Detection format: polygon or bbox
                    if hasattr(detection, "bbox"):
                        bbox_coords = detection.bbox
                    else:
                        # Assume format: [x0, y0, x1, y1] or polygon
                        bbox_coords = detection

                    # Normalize coordinates
                    if isinstance(bbox_coords, (list, tuple)) and len(bbox_coords) >= 4:
                        x0, y0 = bbox_coords[0], bbox_coords[1]
                        x1, y1 = bbox_coords[2], bbox_coords[3]

                        bbox = BBox(
                            x0=x0 / img_w if img_w > 0 else 0.0,
                            y0=y0 / img_h if img_h > 0 else 0.0,
                            x1=x1 / img_w if img_w > 0 else 1.0,
                            y1=y1 / img_h if img_h > 0 else 1.0,
                        )
                        bboxes.append(bbox)
                        scores.append(getattr(detection, "score", 1.0))

                results.append(
                    TextDetection(
                        bboxes=bboxes,
                        texts=[],  # No text yet, only detection
                        scores=scores,
                        page_index=page_idx,
                        dimensions=(img_w, img_h),
                    )
                )

            return results

        except Exception as e:
            raise OCRError(f"Text detection failed: {e}") from e

    def recognize_text(
        self, pages: List[PILImage], detections: List[TextDetection]
    ) -> List[TextDetection]:
        """
        Recognize text in detected regions.

        Args:
            pages: List of PIL images
            detections: List of TextDetection objects from detect_text

        Returns:
            List of TextDetection objects with recognized text
        """
        rasterized_pages = [np.array(page) for page in pages]

        try:
            # Prepare crops for recognition
            all_crops = []
            crop_to_detection = []

            if len(rasterized_pages) != len(detections):
                raise ValueError(f"Mismatch: {len(rasterized_pages)} pages but {len(detections)} detections")
            for page_idx, (page, detection) in enumerate(zip(rasterized_pages, detections)):
                img_h, img_w = page.shape[:2]

                for bbox in detection.bboxes:
                    # Extract crop
                    x0, y0, x1, y1 = bbox.to_absolute(img_w, img_h)
                    crop = page[y0:y1, x0:x1]
                    if crop.size > 0:
                        all_crops.append(Image.fromarray(crop))
                        crop_to_detection.append((page_idx, len(detection.bboxes)))

            # Run recognition
            if not all_crops:
                return detections

            word_preds = self.reco_predictor(all_crops)

            # Map predictions back to detections
            crop_idx = 0
            for page_idx, detection in enumerate(detections):
                texts = []
                for _ in range(len(detection.bboxes)):
                    if crop_idx < len(word_preds):
                        texts.append(str(word_preds[crop_idx]))
                        crop_idx += 1
                    else:
                        texts.append("")

                detection.texts = texts

            return detections

        except Exception as e:
            raise OCRError(f"Text recognition failed: {e}") from e

    def ocr(self, pages: List[PILImage]) -> List[TextDetection]:
        """
        Full OCR pipeline: detection + recognition.

        Args:
            pages: List of PIL images

        Returns:
            List of TextDetection objects with detected and recognized text
        """
        if self.det_predictor is None or self.reco_predictor is None:
            # Return empty detections if OCR not initialized
            return [
                TextDetection(
                    bboxes=[],
                    texts=[],
                    scores=[],
                    page_index=i,
                    dimensions=(page.width, page.height),
                )
                for i, page in enumerate(pages)
            ]
        
        # Step 1: Detect text regions
        detections = self.detect_text(pages)

        # Step 2: Recognize text
        detections = self.recognize_text(pages, detections)

        return detections

    def get_spans_from_detection(self, detection: TextDetection) -> List[Span]:
        """
        Convert TextDetection to list of Spans.

        Args:
            detection: TextDetection object

        Returns:
            List of Span objects
        """
        spans = []
        if len(detection.bboxes) != len(detection.texts) or len(detection.bboxes) != len(detection.scores):
            raise ValueError(f"Mismatch: {len(detection.bboxes)} bboxes, {len(detection.texts)} texts, {len(detection.scores)} scores")
        for bbox, text, score in zip(detection.bboxes, detection.texts, detection.scores):
            spans.append(
                Span(
                    text=text,
                    bbox=bbox,
                    span_type="text",
                    confidence=score,
                )
            )
        return spans

