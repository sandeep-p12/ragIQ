"""YOLO-based layout detection for ParseForge using doclayout_yolo."""

import logging
from pathlib import Path
from typing import List, Optional

import numpy as np
from doclayout_yolo import YOLOv10
from PIL import Image
from PIL.Image import Image as PILImage

from src.config.parsing import ParseForgeConfig
from src.schema.document import BBox, BlockType
from src.utils.exceptions import LayoutError

logger = logging.getLogger(__name__)

# Label mapping for doclayout_yolo_ft.pt model
# Based on model.classes or model.names
LABEL_MAP = {
    0: "title",
    1: "plain text",
    2: "abandon",  # Ignore this class
    3: "figure",
    4: "figure_caption",
    5: "table",
    6: "table_caption",
    7: "table_footnote",
    8: "isolate_formula",
    9: "formula_caption",
}

# Block type mapping from label to our BlockType enum
LABEL_TO_BLOCK_TYPE = {
    0: BlockType.TITLE,              # title
    1: BlockType.TEXT,               # plain text
    2: None,                         # abandon - skip this
    3: BlockType.IMAGE,              # figure
    4: BlockType.CAPTION,            # figure_caption
    5: BlockType.TABLE,              # table
    6: BlockType.CAPTION,            # table_caption
    7: BlockType.FOOTNOTE,           # table_footnote
    8: BlockType.TEXT,                # isolate_formula -> treat as plain text
    9: BlockType.TEXT,                # formula_caption -> treat as plain text
}


class LayoutDetectionOutput:
    """Output from layout detection."""

    def __init__(
        self,
        bbox: BBox,
        category_id: int,
        score: float,
        block_type: BlockType,
    ):
        self.bbox = bbox
        self.category_id = category_id
        self.score = score
        self.block_type = block_type


class YOLOLayoutDetector:
    """YOLO-based layout detector using doclayout_yolo YOLOv10."""

    def __init__(
        self,
        model_path: Optional[Path] = None,
        device: str = "cpu",
        threshold: float = 0.2,
        preserve_aspect_ratio: bool = True,
        config: Optional[ParseForgeConfig] = None,
    ):
        """
        Initialize YOLO layout detector.

        Args:
            model_path: Path to .pt model file (doclayout_yolo_ft.pt)
            device: Device to use (cpu, cuda, mps, coreml)
            threshold: Confidence threshold (default 0.2)
            preserve_aspect_ratio: Whether to preserve aspect ratio
            config: ParseForge configuration
        """
        self.config = config or ParseForgeConfig()
        self.threshold = threshold
        self.preserve_aspect_ratio = preserve_aspect_ratio
        self.imgsz = 1024  # Image size for prediction

        # Determine model path
        # Model folder is at project root, not inside parseforge folder
        if model_path is None:
            model_dir = self.config.model_dir
            
            # If model_dir is relative, resolve it relative to project root (where .env is)
            if not model_dir.is_absolute():
                import os
                # Get project root (where .env file is located)
                # Try to find it by looking for .env or going up from config file
                project_root = Path(os.getcwd())
                # Check if we're in the right directory (look for .env or src/)
                if not (project_root / ".env").exists() and not (project_root / "src").exists():
                    # Try parent directory
                    if (project_root.parent / ".env").exists() or (project_root.parent / "src").exists():
                        project_root = project_root.parent
                model_dir = project_root / model_dir if isinstance(model_dir, str) or str(model_dir).startswith("./") else model_dir
            
            # Look for .pt model first, then fallback to config name
            pt_model = model_dir / "doclayout_yolo_ft.pt"
            if pt_model.exists():
                model_path = pt_model
            else:
                model_name = self.config.yolo_layout_model
                model_path = model_dir / model_name

        if not model_path.exists():
            logger.warning(f"Model file not found: {model_path}. Layout detection will be disabled.")
            logger.info(f"Looking for model in: {model_path.parent}")
            logger.info(f"Model directory exists: {model_path.parent.exists()}")
            if model_path.parent.exists():
                logger.info(f"Files in model directory: {list(model_path.parent.glob('*.pt'))}")
            self.model = None
            return

        try:
            # Load YOLOv10 model from doclayout_yolo
            # According to DocLayout-YOLO repo: model = YOLOv10("path/to/provided/model")
            # Device is passed to predict(), not to the constructor
            logger.info(f"Loading YOLO layout model from {model_path}")
            self.model = YOLOv10(str(model_path))
            
            # Get class names from model
            # Try different ways to access class names based on YOLOv10 structure
            if hasattr(self.model, 'model') and hasattr(self.model.model, 'names'):
                self.class_names = self.model.model.names
            elif hasattr(self.model, 'names'):
                self.class_names = self.model.names
            elif hasattr(self.model, 'model') and hasattr(self.model.model, 'module') and hasattr(self.model.model.module, 'names'):
                self.class_names = self.model.model.module.names
            else:
                # Fallback to default mapping
                self.class_names = LABEL_MAP
                logger.warning("Could not get class names from model, using default mapping")
            
            logger.info(f"Successfully loaded YOLO layout model from {model_path}")
            logger.info(f"Model classes: {self.class_names}")
        except ImportError as e:
            logger.error(f"Failed to import YOLOv10 from doclayout_yolo: {e}")
            logger.error("Make sure 'doclayout-yolo' is installed: pip install doclayout-yolo")
            self.model = None
            self.class_names = LABEL_MAP
        except Exception as e:
            error_type = type(e).__name__
            logger.error(f"Failed to load YOLO model ({error_type}): {e}")
            import traceback
            logger.debug(f"Full traceback: {traceback.format_exc()}")
            self.model = None
            self.class_names = LABEL_MAP

    def _convert_device(self, device: str) -> str:
        """Convert device string to format expected by YOLOv10."""
        if device == "cuda":
            return "cuda:0"
        elif device == "mps":
            return "mps"
        elif device == "coreml":
            return "cpu"  # CoreML not directly supported, use CPU
        else:
            return "cpu"

    def __call__(self, img_pages: List[PILImage]) -> List[List[LayoutDetectionOutput]]:
        """
        Detect layout in a list of page images.

        Args:
            img_pages: List of PIL images

        Returns:
            List of layout detection outputs per page
        """
        if self.model is None:
            # Return empty detections if model not loaded
            return [[] for _ in img_pages]

        results = []
        device_str = self._convert_device(self.config.device)

        for img_page in img_pages:
            try:
                # Convert PIL to numpy array (RGB)
                img_array = np.array(img_page)
                if img_array.ndim != 3:
                    raise LayoutError("Image must be multi-channel 2D image")
                
                img_h, img_w = img_array.shape[:2]

                # YOLOv10.predict according to DocLayout-YOLO repo example:
                # det_res = model.predict("path/to/image", imgsz=1024, conf=0.2, device="cuda:0")
                # It can accept image path, PIL Image, or numpy array
                # Reference: https://github.com/opendatalab/DocLayout-YOLO
                detections = self.model.predict(
                    img_page,           # Can accept PIL Image directly
                    imgsz=self.imgsz,   # Prediction image size
                    conf=self.threshold, # Confidence threshold
                    device=device_str,   # Device to use (e.g., 'cuda:0' or 'cpu')
                    verbose=False,       # Suppress verbose output
                )

                # Extract detections from YOLOv10 results
                # detections is a list of Results objects, one per image
                page_detections = self._extract_bboxes_from_yolo(
                    detections[0], img_h, img_w
                )
                results.append(page_detections)

            except Exception as e:
                logger.error(f"Error in layout detection: {e}")
                import traceback
                logger.debug(f"Full traceback: {traceback.format_exc()}")
                results.append([])

        return results

    def _extract_bboxes_from_yolo(
        self, detection_result, img_h: int, img_w: int
    ) -> List[LayoutDetectionOutput]:
        """Extract bounding boxes from YOLOv10 detection results."""
        results = []

        try:
            # YOLOv10 results have boxes attribute
            if hasattr(detection_result, 'boxes'):
                boxes = detection_result.boxes
                
                # Get box data
                if hasattr(boxes, 'xyxy'):  # xyxy format: [x1, y1, x2, y2]
                    xyxy = boxes.xyxy.cpu().numpy()
                    conf = boxes.conf.cpu().numpy()
                    cls = boxes.cls.cpu().numpy().astype(int)
                elif hasattr(boxes, 'data'):
                    # Alternative format
                    data = boxes.data.cpu().numpy()
                    xyxy = data[:, :4]
                    conf = data[:, 4] if data.shape[1] > 4 else np.ones(len(data))
                    cls = data[:, 5].astype(int) if data.shape[1] > 5 else np.zeros(len(data))
                else:
                    logger.warning("Unexpected boxes format")
                    return []

                for i in range(len(xyxy)):
                    x1, y1, x2, y2 = xyxy[i]
                    score = float(conf[i])
                    category_id = int(cls[i])

                    # Skip if below threshold or if class is "abandon"
                    if score < self.threshold:
                        continue
                    
                    # Skip abandon class (label 2)
                    if category_id == 2:
                        continue

                    # Get block type from mapping
                    block_type = LABEL_TO_BLOCK_TYPE.get(category_id)
                    if block_type is None:
                        # Skip if no mapping (like abandon class)
                        continue

                    # Normalize coordinates to 0-1 range
                    bbox = BBox(
                        x0=x1 / img_w if img_w > 0 else 0.0,
                        y0=y1 / img_h if img_h > 0 else 0.0,
                        x1=x2 / img_w if img_w > 0 else 1.0,
                        y1=y2 / img_h if img_h > 0 else 1.0,
                    )

                    results.append(
                        LayoutDetectionOutput(
                            bbox=bbox,
                            category_id=category_id,
                            score=score,
                            block_type=block_type,
                        )
                    )

            else:
                logger.warning("Detection result does not have boxes attribute")

        except Exception as e:
            logger.error(f"Error extracting bboxes from YOLO results: {e}")
            import traceback
            logger.debug(traceback.format_exc())

        # Sort by score and return
        results.sort(key=lambda x: x.score, reverse=True)
        return results

