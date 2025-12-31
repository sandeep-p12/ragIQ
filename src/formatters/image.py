"""Vision LLM formatter for image descriptions."""

import base64
import logging
from io import BytesIO
from pathlib import Path
from typing import List, Optional, TYPE_CHECKING

from PIL import Image

from src.config.parsing import ParseForgeConfig
from src.config.prompts import IMAGE_DESCRIPTION_PROMPT, PAGE_PROCESSING_PROMPT
from src.providers.llm.openai_llm import OpenAILLMProvider
from src.schema.document import ImageBlock

if TYPE_CHECKING:
    from PIL.Image import Image as PILImage

logger = logging.getLogger(__name__)

IMAGE_DESCRIPTION_TAG = "IMAGE_DESCRIPTION"


class ImageVisionLLMFormatter:
    """Generate image descriptions using vision-capable LLM."""

    def __init__(self, config: Optional[ParseForgeConfig] = None):
        """
        Initialize image vision LLM formatter.

        Args:
            config: ParseForge configuration
        """
        self.config = config or ParseForgeConfig()
        self.llm_provider = OpenAILLMProvider(config)

    def _image_to_base64(self, image: Image.Image) -> str:
        """Convert PIL Image to base64 string."""
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        return img_str

    def _load_image(self, image_block: ImageBlock) -> Optional[Image.Image]:
        """Load image from ImageBlock."""
        if image_block.image_data:
            return Image.open(BytesIO(image_block.image_data))
        elif image_block.image_path:
            path = Path(image_block.image_path)
            if path.exists():
                return Image.open(path)
        return None

    def _is_important_image(self, image_block: ImageBlock, image: "Image.Image") -> bool:
        """
        Determine if image is important (KPI, charts, flowcharts) vs icon/decorative.
        
        Args:
            image_block: Image block
            image: PIL Image object
            
        Returns:
            True if image is important, False if decorative/icon
        """
        # Check image size - icons are usually small
        width, height = image.size
        if width < 100 or height < 100:
            logger.debug(f"Image too small ({width}x{height}), likely icon")
            return False
        
        # Check aspect ratio - icons are often square or very small
        aspect_ratio = width / height if height > 0 else 1
        if 0.8 < aspect_ratio < 1.2 and (width < 200 or height < 200):
            logger.debug(f"Small square image ({width}x{height}), likely icon")
            return False
        
        # If image has caption, it's likely important
        if image_block.caption and len(image_block.caption.strip()) > 10:
            return True
        
        # If image has alt text suggesting importance
        if image_block.alt_text:
            alt_lower = image_block.alt_text.lower()
            important_keywords = ["chart", "graph", "diagram", "flowchart", "kpi", "metric", 
                                 "table", "figure", "illustration", "visualization"]
            if any(keyword in alt_lower for keyword in important_keywords):
                return True
        
        # Default: assume important if large enough
        return True

    def describe_image(self, image_block: ImageBlock) -> str:
        """
        Generate description for a single image.
        Only processes important images (KPI, charts, flowcharts), skips icons.

        Args:
            image_block: Image block to describe

        Returns:
            Image description string, or empty string if failed/skipped
        """
        if self.llm_provider.client is None:
            logger.warning("Vision LLM client not available")
            return image_block.alt_text or image_block.caption or "Image"

        # Load image
        image = self._load_image(image_block)
        if image is None:
            logger.warning(f"Could not load image from block {image_block.block_id}")
            return image_block.alt_text or image_block.caption or "Image"

        # Check if image is important
        if not self._is_important_image(image_block, image):
            logger.debug(f"Skipping image {image_block.block_id} - appears to be icon/decorative")
            return ""  # Return empty to skip in markdown

        try:
            if self.config.llm_provider == "openai":
                return self._describe_with_openai(image, image_block)
            else:
                return image_block.alt_text or image_block.caption or "Image"
        except Exception as e:
            logger.error(f"Failed to generate image description: {e}")
            return image_block.alt_text or image_block.caption or "Image"

    def _describe_with_openai(self, image: "PILImage", image_block: ImageBlock) -> str:
        """Describe image using OpenAI vision model."""
        # Create prompt - focus on important content
        prompt = IMAGE_DESCRIPTION_PROMPT

        # Ensure model supports vision (fallback to gpt-4o if not specified)
        model = self.config.llm_model
        if model not in ["gpt-4o", "gpt-4-vision-preview", "gpt-4-turbo"]:
            # Try to use a vision-capable model
            if "gpt-4" in model.lower():
                model = "gpt-4o"  # Default to gpt-4o for vision
            else:
                logger.warning(f"Model {model} may not support vision. Using gpt-4o.")
                model = "gpt-4o"

        try:
            # Call LLM via provider with vision
            description = self.llm_provider.generate_vision(
                prompt=prompt,
                images=[image],
                model=model,
                temperature=0.0,
                max_tokens=500,
            ).strip()
            return description
        except Exception as e:
            logger.error(f"OpenAI vision API error: {e}")
            # Fallback to alt text or caption
            return image_block.alt_text or image_block.caption or "Image"

    def describe_images_batch(self, image_blocks: List[ImageBlock]) -> List[str]:
        """
        Generate descriptions for multiple images (batch processing).

        Args:
            image_blocks: List of image blocks

        Returns:
            List of descriptions (one per image block)
        """
        descriptions = []
        for image_block in image_blocks:
            description = self.describe_image(image_block)
            descriptions.append(description)
        return descriptions

    def format_description_tag(self, description: str) -> str:
        """
        Format image description with identifier tag.

        Args:
            description: Image description text

        Returns:
            Formatted tag string: [IMAGE_DESCRIPTION: <description>]
        """
        return f"[{IMAGE_DESCRIPTION_TAG}: {description}]"

    def process_page_with_images(
        self,
        page_image: "PILImage",
        image_blocks: List[ImageBlock],
        page_index: int,
    ) -> dict:
        """
        Process a whole page through vision LLM to get OCR text and image descriptions.
        This maintains layout and positions.

        Args:
            page_image: Full page image
            image_blocks: List of image blocks detected on the page
            page_index: Page index

        Returns:
            Dictionary with:
            - 'ocr_markdown': OCR text formatted as markdown
            - 'image_descriptions': Dict mapping image block_id to description
            - 'blocks': List of blocks with text filled from OCR
        """
        if self.llm_provider.client is None:
            logger.warning("Vision LLM client not available for page processing")
            return {
                'ocr_markdown': '',
                'image_descriptions': {},
                'blocks': []
            }

        try:
            # Create comprehensive prompt - LLM handles everything
            # The LLM maintains layout, positions, OCR, and image descriptions all at once
            prompt = PAGE_PROCESSING_PROMPT

            # Ensure model supports vision
            model = self.config.llm_model
            if model not in ["gpt-4o", "gpt-4-vision-preview", "gpt-4-turbo"]:
                if "gpt-4" in model.lower():
                    model = "gpt-4o"
                else:
                    logger.warning(f"Model {model} may not support vision. Using gpt-4o.")
                    model = "gpt-4o"

            # Call LLM via provider with vision
            ocr_markdown = self.llm_provider.generate_vision(
                prompt=prompt,
                images=[page_image],
                model=model,
                temperature=0.0,
                max_tokens=4000,
            ).strip()

            # Extract image descriptions from markdown
            image_descriptions = {}
            import re
            
            # Find all image description tags in the markdown
            pattern = r'\[IMAGE_DESCRIPTION:\s*(.*?)\]'
            matches = re.finditer(pattern, ocr_markdown, re.DOTALL)
            
            # Map descriptions to image blocks by position (approximate)
            # We'll match descriptions to images based on order and position
            descriptions = [match.group(1).strip() for match in matches]
            
            # Filter to important images only
            important_images = []
            for img_block in image_blocks:
                # Check if image is important
                if img_block.image_data:
                    img = Image.open(BytesIO(img_block.image_data))
                    if self._is_important_image(img_block, img):
                        important_images.append(img_block)
            
            # Assign descriptions to images (by order)
            for i, img_block in enumerate(important_images):
                if i < len(descriptions):
                    image_descriptions[img_block.block_id] = descriptions[i]
                else:
                    # Generate individual description if not in OCR result
                    desc = self.describe_image(img_block)
                    if desc:
                        image_descriptions[img_block.block_id] = desc

            # Remove image description tags from markdown (they'll be inserted at block positions)
            ocr_markdown = re.sub(r'\[IMAGE_DESCRIPTION:.*?\]', '', ocr_markdown, flags=re.DOTALL)

            return {
                'ocr_markdown': ocr_markdown,
                'image_descriptions': image_descriptions,
                'blocks': []  # Will be populated by caller
            }

        except Exception as e:
            logger.error(f"Failed to process page {page_index} with vision LLM: {e}")
            import traceback
            logger.debug(f"Page processing error: {traceback.format_exc()}")
            return {
                'ocr_markdown': '',
                'image_descriptions': {},
                'blocks': []
            }

