"""PDF parser for ParseForge."""

import logging
from collections import Counter
from pathlib import Path
from typing import Callable, Dict, List, Optional

import numpy as np
import pypdfium2 as pdfium
from PIL import Image

from src.config.parsing import ParseForgeConfig
from src.config.parsing_strategies import PageStrategy, StrategyEnum, determine_global_strategy, get_page_strategy
from src.pipelines.parsing.parsers.block_processor import (
    fill_text_from_native,
    fill_text_from_ocr,
    layout_detections_to_blocks,
)
from src.formatters.image import ImageVisionLLMFormatter
from src.pipelines.parsing.parsers.native_pdf_extractor import extract_blocks_from_native_pdf
from src.providers.ocr.doctr import DoctrOCR, TextDetection
from src.pipelines.parsing.processing.magic import MagicModel
from src.pipelines.parsing.processing.reading_order import sort_blocks_by_reading_order
from src.pipelines.parsing.processing.para_split import split_paragraphs
from src.pipelines.parsing.processing.table_extractor import extract_table
from src.pipelines.parsing.processing.table_merger import merge_cross_page_tables
from src.providers.layout.yolo import YOLOLayoutDetector
from src.config.prompts import BASE_LLM_PROMPT
from src.schema.document import BBox, Block, BlockType, Document, ImageBlock, Page, TableBlock, TextBlock, TitleBlock
from src.utils.exceptions import ParserError

logger = logging.getLogger(__name__)


class PDFParser:
    """PDF parser with AUTO strategy selection."""

    def __init__(self, config: Optional[ParseForgeConfig] = None, progress_callback: Optional[Callable[[str, float, Optional[Dict]], None]] = None):
        """
        Initialize PDF parser.

        Args:
            config: ParseForge configuration
            progress_callback: Optional progress callback (stage, progress, output_data)
        """
        self.config = config or ParseForgeConfig()
        self.progress_callback = progress_callback
        try:
            self.doctr_ocr = DoctrOCR(
                device=self.config.device,
                det_arch=self.config.doctr_det_arch,
                reco_arch=self.config.doctr_reco_arch,
                config=self.config,
            )
        except Exception as e:
            logger.warning(f"Failed to initialize Doctr OCR: {e}. OCR will be disabled.")
            self.doctr_ocr = None
        try:
            self.layout_detector = YOLOLayoutDetector(
                device=self.config.device,
                config=self.config,
            )
        except Exception as e:
            logger.warning(f"Failed to initialize layout detector: {e}. Layout detection will be disabled.")
            self.layout_detector = None
        
        # Initialize vision LLM formatter for whole-page processing
        try:
            self.vision_llm = ImageVisionLLMFormatter(config=self.config)
        except Exception as e:
            logger.warning(f"Failed to initialize vision LLM formatter: {e}")
            self.vision_llm = None

    def parse(
        self,
        file_path: str,
        strategy: StrategyEnum = StrategyEnum.AUTO,
        start_page: int = 0,
        end_page: Optional[int] = None,
    ) -> Document:
        """
        Parse PDF file.

        Args:
            file_path: Path to PDF file
            strategy: Parsing strategy
            start_page: Start page index (0-based)
            end_page: End page index (None for all pages)

        Returns:
            Document object
        """
        # Handle LLM_FULL strategy separately - bypass all normal parsing
        if strategy == StrategyEnum.LLM_FULL:
            logger.info(f"Using LLM_FULL strategy - bypassing layout detection, OCR, and table extraction")
            return self._parse_with_llm_full(file_path, start_page, end_page)
        
        try:
            pdf_doc = pdfium.PdfDocument(file_path)
            total_pages = len(pdf_doc)

            if end_page is None:
                end_page = total_pages

            pages = []
            page_strategies = []

            # Rasterize pages for strategy selection
            rasterized_pages = []
            for page_idx in range(start_page, min(end_page, total_pages)):
                page = pdf_doc[page_idx]
                img = page.render(scale=2.0).to_pil()
                rasterized_pages.append((page_idx, page, img))

            # Determine strategy for each page
            if strategy == StrategyEnum.AUTO:
                # Run OCR detection on all pages
                if self.doctr_ocr is not None:
                    images = [img for _, _, img in rasterized_pages]
                    detections = self.doctr_ocr.detect_text(images)
                else:
                    # If OCR not available, default to FAST for all pages
                    logger.warning("OCR not available, defaulting to FAST strategy for all pages")
                    detections = [None] * len(rasterized_pages)

                if len(rasterized_pages) != len(detections):
                    raise ValueError(f"Mismatch: {len(rasterized_pages)} pages but {len(detections)} detections")
                for (page_idx, pdf_page, _), detection in zip(rasterized_pages, detections):
                    if detection is not None:
                        page_strategy = get_page_strategy(
                            pdf_page,
                            detection,
                            threshold=self.config.get_strategy_config().page_threshold,
                        )
                    else:
                        # Default to FAST if detection failed
                        page_strategy = StrategyEnum.FAST
                    page_strategies.append(PageStrategy(page_idx, page_strategy))

                # Determine global strategy
                global_strategy = determine_global_strategy(
                    page_strategies,
                    document_threshold=self.config.get_strategy_config().document_threshold,
                )
            else:
                global_strategy = strategy
                for page_idx, _, _ in rasterized_pages:
                    page_strategies.append(PageStrategy(page_idx, global_strategy))

            # Process pages in batches
            # Use configured batch size directly
            batch_size = self.config.batch_size

            for batch_start in range(0, len(rasterized_pages), batch_size):
                batch_end = min(batch_start + batch_size, len(rasterized_pages))
                batch_pages = rasterized_pages[batch_start:batch_end]

                # Process batch
                batch_images = [img for _, _, img in batch_pages]
                batch_page_objects = [(idx, page, img) for idx, page, img in batch_pages]

                # Layout detection - ALWAYS use for structure detection (images, tables)
                # This is critical even for FAST strategy to detect images and tables
                if self.layout_detector is not None:
                    layout_results = self.layout_detector(batch_images)
                else:
                    # Return empty detections if layout detector not available
                    logger.warning("Layout detector not available - images and tables may not be detected properly")
                    layout_results = [[] for _ in batch_images]

                # Process each page in batch
                if len(batch_page_objects) != len(layout_results):
                    raise ValueError(f"Mismatch: {len(batch_page_objects)} pages but {len(layout_results)} layout results")
                for (page_idx, pdf_page, img), layout_dets in zip(batch_page_objects, layout_results):
                    page_strategy = next(
                        (ps for ps in page_strategies if ps.page_index == page_idx),
                        PageStrategy(page_idx, global_strategy),
                    )

                    page_width = int(pdf_page.get_width())
                    page_height = int(pdf_page.get_height())

                    # Create blocks from layout detection (PREFERRED) or native PDF extraction (FALLBACK)
                    # Layout detection is critical for detecting images and tables accurately
                    if layout_dets:
                        # Post-process layout detections
                        magic_model = MagicModel(layout_detections=layout_dets)
                        cleaned_dets = magic_model.get_detections()

                        # Convert to blocks
                        blocks = layout_detections_to_blocks(
                            cleaned_dets,
                            page_width,
                            page_height,
                            page_idx,
                        )
                    else:
                        # No layout detection available - extract from native PDF as fallback
                        logger.info(f"Layout detection not available for page {page_idx}, using native PDF extraction")
                        native_blocks = extract_blocks_from_native_pdf(
                            pdf_page,
                            page_width,
                            page_height,
                            page_idx,
                        )
                        
                        # Convert to block objects
                        blocks = []
                        for block_type, bbox, text in native_blocks:
                            if block_type == BlockType.TITLE:
                                blocks.append(TitleBlock(
                                    block_type=BlockType.TITLE,
                                    bbox=bbox,
                                    page_index=page_idx,
                                    text=text,
                                    level=1,
                                ))
                            elif block_type == BlockType.TABLE:
                                blocks.append(TableBlock(
                                    block_type=BlockType.TABLE,
                                    bbox=bbox,
                                    page_index=page_idx,
                                ))
                            elif block_type == BlockType.IMAGE:
                                blocks.append(ImageBlock(
                                    block_type=BlockType.IMAGE,
                                    bbox=bbox,
                                    page_index=page_idx,
                                ))
                            else:
                                blocks.append(TextBlock(
                                    block_type=BlockType.TEXT,
                                    bbox=bbox,
                                    page_index=page_idx,
                                    text=text,
                                ))
                    
                    # Extract image data from PDF for ImageBlocks
                    # Extract images from the rendered page image by cropping detected regions
                    image_blocks = [b for b in blocks if isinstance(b, ImageBlock)]
                    if image_blocks:
                        try:
                            from io import BytesIO
                            
                            for i, img_block in enumerate(image_blocks):
                                if img_block.bbox:
                                    # Crop image region from page image
                                    img_x0, img_y0, img_x1, img_y1 = img_block.bbox.to_absolute(page_width, page_height)
                                    
                                    # Ensure coordinates are within image bounds
                                    img_width, img_height = img.size
                                    img_x0 = max(0, int(img_x0))
                                    img_y0 = max(0, int(img_y0))
                                    img_x1 = min(img_width, int(img_x1))
                                    img_y1 = min(img_height, int(img_y1))
                                    
                                    if img_x1 > img_x0 and img_y1 > img_y0:
                                        # Crop image region
                                        image_crop = img.crop((img_x0, img_y0, img_x1, img_y1))
                                        
                                        # Convert to bytes
                                        img_bytes = BytesIO()
                                        image_crop.save(img_bytes, format="PNG")
                                        img_bytes.seek(0)
                                        
                                        # Update image block with extracted data
                                        updated_block = img_block.model_copy(update={
                                            "image_data": img_bytes.getvalue()
                                        })
                                        
                                        # Replace in blocks list
                                        block_idx = blocks.index(image_blocks[i])
                                        blocks[block_idx] = updated_block
                                        
                                        logger.debug(f"Extracted image data for block {img_block.block_id} ({img_x1-img_x0}x{img_y1-img_y0})")
                        except Exception as e:
                            logger.warning(f"Failed to extract images from page {page_idx}: {e}")
                            import traceback
                            logger.debug(f"Image extraction error: {traceback.format_exc()}")
                    
                    # If page has images, process whole page through vision LLM for OCR and image descriptions
                    # This maintains layout and positions
                    page_has_images = len(image_blocks) > 0
                    vision_llm_processed = False
                    if page_has_images and self.vision_llm is not None and self.vision_llm.llm_provider.client is not None:
                        try:
                            logger.info(f"Processing page {page_idx} with images through vision LLM")
                            if self.progress_callback:
                                # Calculate progress - need to get total_pages from outer scope
                                # For now, use a reasonable estimate
                                progress = (page_idx + 1) / max(1, end_page - start_page) if end_page else (page_idx + 1) / max(1, total_pages - start_page)
                                self.progress_callback(
                                    f"Processing page {page_idx + 1} with vision LLM",
                                    progress,
                                    {"page": page_idx + 1, "total": total_pages}
                                )
                            
                            vision_llm_result = self.vision_llm.process_page_with_images(
                                page_image=img,
                                image_blocks=image_blocks,
                                page_index=page_idx,
                            )
                            
                            if vision_llm_result and vision_llm_result.get('ocr_markdown'):
                                # Store the LLM-generated markdown directly
                                # The markdown already maintains layout, positions, and image descriptions
                                # We'll use it directly in to_markdown() instead of converting blocks
                                vision_llm_markdown = vision_llm_result['ocr_markdown']
                                vision_llm_processed = True
                                logger.info(f"Processed page {page_idx} with vision LLM - markdown length: {len(vision_llm_markdown)}")
                                
                                # Create a placeholder page that will store the markdown in metadata
                                # We'll handle this specially in to_markdown()
                                # For now, create empty blocks - markdown will be in page metadata
                                blocks = []  # Empty blocks - markdown stored separately
                        except Exception as e:
                            logger.warning(f"Failed to process page {page_idx} with vision LLM: {e}")
                            import traceback
                            logger.debug(f"Vision LLM processing error: {traceback.format_exc()}")
                            # Continue with normal processing

                    # Fill text content for text blocks (skip if vision LLM already processed)
                    if not vision_llm_processed:
                        # Always fill text - use OCR for HI_RES strategy, native extraction for FAST strategy
                        if page_strategy.strategy == StrategyEnum.HI_RES and self.doctr_ocr is not None:
                            # Use OCR for scanned documents
                            try:
                                ocr_detection = self.doctr_ocr.ocr([img])[0]
                                blocks = fill_text_from_ocr(blocks, ocr_detection, page_width, page_height)
                                
                                # Check if OCR filled enough text - if not, fallback to native extraction
                                text_blocks = [b for b in blocks if isinstance(b, (TextBlock, TitleBlock))]
                                filled_count = sum(1 for b in text_blocks if b.text and b.text.strip())
                                fill_ratio = filled_count / len(text_blocks) if text_blocks else 0
                                
                                # If less than 50% of blocks have text, try native extraction as fallback
                                if fill_ratio < 0.5:
                                    logger.debug(f"OCR only filled {fill_ratio:.1%} of blocks for page {page_idx}, trying native extraction as fallback")
                                    blocks = fill_text_from_native(blocks, pdf_page, page_width, page_height)
                            except Exception as e:
                                logger.warning(f"OCR failed for page {page_idx}: {e}. Falling back to native text.")
                                blocks = fill_text_from_native(blocks, pdf_page, page_width, page_height)
                        else:
                            # Use native text extraction for native PDFs (FAST strategy)
                            blocks = fill_text_from_native(blocks, pdf_page, page_width, page_height)

                    # Extract tables (skip if vision LLM already processed)
                    if not vision_llm_processed:
                        table_blocks = [b for b in blocks if isinstance(b, TableBlock)]
                        text_blocks = [b for b in blocks if isinstance(b, (TextBlock, TitleBlock))]

                        for i, table_block in enumerate(table_blocks):
                            if table_block.bbox:
                                try:
                                    # Pass page image, OCR engine, and config for table extraction
                                    # Fallback order: text blocks -> OCR -> Vision LLM
                                    extracted_table = extract_table(
                                        table_block.bbox,
                                        text_blocks,
                                        page_width,
                                        page_height,
                                        page_image=img,  # Pass page image for OCR/LLM fallback
                                        ocr_engine=self.doctr_ocr if self.doctr_ocr is not None else None,
                                        config=self.config,  # Pass config for vision LLM fallback
                                    )
                                    # Create updated table block with extracted data
                                    updated_table = table_block.model_copy(update={
                                        "html": extracted_table.html,
                                        "cells": extracted_table.cells,
                                        "num_rows": extracted_table.num_rows,
                                        "num_cols": extracted_table.num_cols,
                                        "page_index": page_idx,
                                    })
                                    # Replace in blocks list
                                    block_idx = blocks.index(table_blocks[i])
                                    blocks[block_idx] = updated_table
                                except Exception as e:
                                    logger.warning(f"Failed to extract table: {e}")

                    # Apply reading order (always apply to maintain order)
                    blocks = sort_blocks_by_reading_order(blocks, page_width, page_height)

                    # Paragraph splitting
                    text_title_blocks = [b for b in blocks if isinstance(b, (TextBlock, TitleBlock))]
                    if text_title_blocks:
                        processed_blocks = split_paragraphs(text_title_blocks)
                        # Replace in blocks list using block_id matching
                        processed_dict = {b.block_id: b for b in processed_blocks}
                        for i, orig_block in enumerate(blocks):
                            if isinstance(orig_block, (TextBlock, TitleBlock)) and orig_block.block_id in processed_dict:
                                blocks[i] = processed_dict[orig_block.block_id]

                    # Create page object
                    # If vision LLM processed, store markdown in page metadata
                    page_metadata = {}
                    if vision_llm_processed and vision_llm_result and vision_llm_result.get('ocr_markdown'):
                        page_metadata['vision_llm_markdown'] = vision_llm_result['ocr_markdown']
                        page_metadata['vision_llm_processed'] = True
                    
                    page_obj = Page(
                        page_index=page_idx,
                        width=page_width,
                        height=page_height,
                        blocks=blocks,
                        metadata=page_metadata,
                    )
                    pages.append(page_obj)

            # Merge cross-page tables
            all_table_blocks = []
            for page in pages:
                all_table_blocks.extend([b for b in page.blocks if isinstance(b, TableBlock)])
            
            if len(all_table_blocks) > 1:
                merged_tables = merge_cross_page_tables(all_table_blocks)
                # Update pages with merged tables
                table_idx = 0
                for i, page in enumerate(pages):
                    updated_blocks = []
                    for block in page.blocks:
                        if isinstance(block, TableBlock):
                            if table_idx < len(merged_tables):
                                updated_blocks.append(merged_tables[table_idx])
                                table_idx += 1
                        else:
                            updated_blocks.append(block)
                    # Update page blocks
                    pages[i] = page.model_copy(update={"blocks": updated_blocks})

            # Collect label statistics for UI display
            if self.layout_detector is not None and hasattr(self.layout_detector, 'class_names') and self.progress_callback:
                from collections import Counter
                block_type_counts = Counter()
                for page in pages:
                    for block in page.blocks:
                        block_type_counts[block.block_type.value] += 1
                
                model_classes = self.layout_detector.class_names.copy()
                
                self.progress_callback(
                    "layout_detection",
                    1.0,
                    {
                        "pages_processed": len(pages),
                        "counts": dict(block_type_counts),
                        "model_classes": model_classes,
                    }
                )

            # Create document
            document = Document(
                file_path=file_path,
                file_name=Path(file_path).name,
                pages=pages,
                total_pages=len(pages),
            )

            return document

        except Exception as e:
            raise ParserError(f"Failed to parse PDF: {e}") from e

    def _parse_with_llm_full(
        self,
        file_path: str,
        start_page: int = 0,
        end_page: Optional[int] = None,
    ) -> Document:
        """
        Parse entire document using LLM vision API.
        Similar to MegaParse vision approach - processes all pages through LLM.

        Args:
            file_path: Path to PDF file
            start_page: Start page index (0-based)
            end_page: End page index (None for all pages)

        Returns:
            Document object
        """
        if self.vision_llm is None or self.vision_llm.llm_provider.client is None:
            raise ParserError("LLM_FULL strategy requires LLM configuration. Set PARSEFORGE_LLM_PROVIDER=openai and PARSEFORGE_LLM_API_KEY in .env file.")

        try:
            import base64
            from io import BytesIO

            pdf_doc = pdfium.PdfDocument(file_path)
            total_pages = len(pdf_doc)

            if end_page is None:
                end_page = total_pages

            # Convert pages to base64 images
            page_images_base64 = []
            page_dimensions = []
            
            if self.progress_callback:
                self.progress_callback("Converting PDF to images", 0.1, {"total_pages": end_page - start_page})

            for page_idx in range(start_page, min(end_page, total_pages)):
                page = pdf_doc[page_idx]
                img = page.render(scale=2.0).to_pil()
                
                # Convert to base64
                buffered = BytesIO()
                img.save(buffered, format="PNG")
                img_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
                page_images_base64.append(img_base64)
                page_dimensions.append((int(page.get_width()), int(page.get_height())))

            # Process pages in batches
            batch_size = min(3, len(page_images_base64))  # LLM typically handles 3 pages at a time
            parsed_chunks = []
            

            total_batches = (len(page_images_base64) + batch_size - 1) // batch_size
            
            for batch_idx in range(0, len(page_images_base64), batch_size):
                batch_end = min(batch_idx + batch_size, len(page_images_base64))
                batch_images = page_images_base64[batch_idx:batch_end]
                
                if self.progress_callback:
                    progress = 0.2 + (batch_idx / len(page_images_base64)) * 0.7
                    self.progress_callback(
                        f"Processing pages {batch_idx + 1}-{batch_end} with LLM",
                        progress,
                        {"batch": batch_idx // batch_size + 1, "total_batches": total_batches}
                    )

                # Calculate actual page numbers (1-based) for this batch
                batch_start_page_1based = start_page + batch_idx + 1
                batch_end_page_1based = start_page + batch_end
                
                # Create batch-specific prompt with page information
                if len(batch_images) == 1:
                    page_info = f"\n\nIMPORTANT: You are processing PAGE {batch_start_page_1based}. Add the page identifier '--- Page {batch_start_page_1based} ---' at the start of your output."
                else:
                    # List all pages in the batch for individual identifiers
                    page_list = ", ".join([f"--- Page {start_page + batch_idx + j + 1} ---" for j in range(len(batch_images))])
                    page_info = f"\n\nIMPORTANT: You are processing {len(batch_images)} pages ({batch_start_page_1based} through {batch_end_page_1based}). Add individual page identifiers at the start of content from each page:\n- Page {batch_start_page_1based}: '--- Page {batch_start_page_1based} ---'\n- Page {batch_start_page_1based + 1}: '--- Page {batch_start_page_1based + 1} ---'"
                    if len(batch_images) > 2:
                        page_info += f"\n- And so on for each page (Page {batch_start_page_1based + 2}, Page {batch_end_page_1based}, etc.)"
                    page_info += "\nEach page identifier should be on its own line with blank lines before and after."
                
                batch_prompt = BASE_LLM_PROMPT + page_info

                # Call OpenAI vision API
                try:
                    model = self.config.llm_model
                    if model not in ["gpt-4o", "gpt-4-vision-preview", "gpt-4-turbo"]:
                        if "gpt-4" in model.lower():
                            model = "gpt-4o"
                        else:
                            logger.warning(f"Model {model} may not support vision. Using gpt-4o.")
                            model = "gpt-4o"

                    # Use the LLM provider's generate_vision method
                    # Pass base64 strings directly (not formatted dicts)
                    chunk_content = self.vision_llm.llm_provider.generate_vision(
                        prompt=batch_prompt,
                        images=batch_images,  # Pass base64 strings directly
                        model=model,
                        temperature=0.1,
                        max_tokens=8000,
                    ).strip()
                    parsed_chunks.append(chunk_content)
                    logger.info(f"Processed batch {batch_idx // batch_size + 1}/{total_batches} - {len(chunk_content)} chars")

                except Exception as e:
                    logger.error(f"Failed to process batch {batch_idx // batch_size + 1}: {e}")
                    # Continue with empty chunk
                    parsed_chunks.append("")

            # Combine all chunks
            full_content = "\n".join(parsed_chunks)
            
            if self.progress_callback:
                self.progress_callback("Processing LLM response", 0.9, {"content_length": len(full_content)})

            # Process content - preserve tags but clean formatting
            processed_content = self._process_llm_content(full_content)

            # Convert to Document structure
            # Create a single page with the full content as a text block
            from src.schema.document import TextBlock, BBox, Page, Document
            
            # Ensure we have valid dimensions
            page_width = page_dimensions[0][0] if page_dimensions and len(page_dimensions) > 0 else 800
            page_height = page_dimensions[0][1] if page_dimensions and len(page_dimensions) > 0 else 1000
            
            # Create a text block covering the entire document
            # Limit text length to avoid validation issues (store full content in metadata)
            text_preview = processed_content[:1000] if len(processed_content) > 1000 else processed_content
            full_text_block = TextBlock(
                block_type=BlockType.TEXT,
                bbox=BBox(x0=0.0, y0=0.0, x1=1.0, y1=1.0),
                page_index=0,
                text=text_preview,  # Store preview in block, full content in metadata
            )

            # Create a single page containing the full document
            # Store the markdown in page metadata so to_markdown() can return it directly
            page_obj = Page(
                page_index=0,
                width=page_width,
                height=page_height,
                blocks=[full_text_block],  # Keep block for compatibility, but markdown is in metadata
                metadata={
                    "llm_full_processed": True,
                    "llm_full_markdown": processed_content,  # Store complete markdown here
                },
            )

            # Create document
            # Store actual page range information for proper page numbering
            # Pages processed: range(start_page, min(end_page, total_pages))
            # Convert to 1-based for display
            actual_start_page = start_page + 1  # Convert to 1-based
            # The last page index processed is min(end_page, total_pages) - 1 (0-based)
            # In 1-based, that's min(end_page, total_pages)
            actual_end_page = min(end_page, total_pages)  # 1-based (inclusive)
            
            # Create document with proper validation
            # Use model_validate to ensure proper Pydantic validation
            try:
                document_dict = {
                    "file_path": file_path,
                    "file_name": Path(file_path).name,
                    "pages": [page_obj.model_dump()],  # Convert to dict for validation
                    "total_pages": 1,
                    "metadata": {
                        "parsing_strategy": "llm_full",
                        "llm_full_start_page": actual_start_page,
                        "llm_full_end_page": actual_end_page,
                        "llm_full_total_pages": actual_end_page - actual_start_page + 1,
                    },
                }
                document = Document.model_validate(document_dict)
            except Exception as e:
                logger.error(f"Error creating Document with model_validate: {e}")
                # Fallback to direct instantiation
                try:
                    document = Document(
                        file_path=file_path,
                        file_name=Path(file_path).name,
                        pages=[page_obj],
                        total_pages=1,
                        metadata={
                            "parsing_strategy": "llm_full",
                            "llm_full_start_page": actual_start_page,
                            "llm_full_end_page": actual_end_page,
                            "llm_full_total_pages": actual_end_page - actual_start_page + 1,
                        },
                    )
                except Exception as e2:
                    logger.error(f"Error creating Document with direct instantiation: {e2}")
                    logger.error(f"Page object type: {type(page_obj)}")
                    logger.error(f"Page object: {page_obj}")
                    if hasattr(page_obj, 'blocks'):
                        logger.error(f"Page blocks count: {len(page_obj.blocks)}")
                        if page_obj.blocks:
                            logger.error(f"First block type: {type(page_obj.blocks[0])}")
                    raise ParserError(f"Failed to create Document object: {e2}") from e2

            if self.progress_callback:
                self.progress_callback("LLM parsing complete", 1.0, {"total_chars": len(processed_content)})

            return document

        except Exception as e:
            raise ParserError(f"Failed to parse PDF with LLM_FULL strategy: {e}") from e

    def _process_llm_content(self, content: str) -> str:
        """
        Process LLM response content while preserving identifier tags.
        Formats tables, adds proper spacing, and improves readability.
        Tables are protected from regex substitutions that might corrupt them.

        Args:
            content: Raw LLM response with tags

        Returns:
            Processed content with tags preserved and properly formatted
        """
        import re

        # Helper function to sanitize table cell content
        def sanitize_cell(cell: str) -> str:
            """Sanitize cell content: remove pipes, normalize whitespace."""
            if not cell:
                return ""
            # Replace pipe characters with space (pipes break markdown tables)
            cell = cell.replace('|', ' ')
            # Normalize whitespace
            cell = re.sub(r'\s+', ' ', cell.strip())
            return cell

        # Step 1: Extract and protect tables from regex substitutions
        table_placeholders = {}
        table_counter = 0
        
        def extract_table(match):
            nonlocal table_counter
            table_content = match.group(1)
            placeholder = f"__TABLE_PLACEHOLDER_{table_counter}__"
            table_placeholders[placeholder] = table_content
            table_counter += 1
            return placeholder
        
        # Extract all tables with [TABLE] tags first
        table_pattern = r"\[TABLE\](.*?)\[/TABLE\]"
        content = re.sub(table_pattern, extract_table, content, flags=re.DOTALL)
        
        # Also extract raw markdown tables (consecutive lines starting and ending with |)
        # More precise: match table-like structures (at least 2 rows, with separator row)
        def extract_raw_table(match):
            nonlocal table_counter
            table_content = match.group(0)
            # Verify it looks like a table (has separator row with ---)
            lines = table_content.strip().split('\n')
            has_separator = any('---' in line or re.match(r'^\|[\s\-:]+\|', line) for line in lines)
            if has_separator and len(lines) >= 2:
                placeholder = f"__TABLE_PLACEHOLDER_{table_counter}__"
                table_placeholders[placeholder] = table_content
                table_counter += 1
                return placeholder
            return match.group(0)  # Return original if not a valid table
        
        # Pattern: consecutive lines that start and end with |
        # Must have at least 2 rows and look like a table
        raw_table_pattern = r"(?:^\|[^\n]*\|(?:\r?\n|$)){2,}"
        content = re.sub(raw_table_pattern, extract_raw_table, content, flags=re.MULTILINE)

        # Handle HEADER tag specially - keep only first occurrence but preserve tag
        header_pattern = r"\[HEADER\](.*?)\[/HEADER\]"
        headers = re.findall(header_pattern, content, re.DOTALL)
        if headers:
            first_header_content = headers[0].strip()
            # Remove all HEADER tags and their content
            content = re.sub(header_pattern, "", content, flags=re.DOTALL)
            # Add the first header back at the beginning with tag preserved
            content = f"[HEADER]\n{first_header_content}\n[/HEADER]\n\n{content}"

        # Clean up markdown artifacts while preserving tags
        # Remove code block markers that might wrap the content
        content = re.sub(r"^```(?:markdown|md)?\s*\n", "", content, flags=re.MULTILINE)
        content = re.sub(r"\n```\s*$", "", content, flags=re.MULTILINE)
        
        # Ensure proper spacing around tags (add blank lines) - but not for placeholders
        content = re.sub(r"(\S)\[IMAGE\]", r"\1\n\n[IMAGE]", content)
        content = re.sub(r"\[/IMAGE\](\S)", r"[/IMAGE]\n\n\1", content)
        content = re.sub(r"(\S)\[TOC\]", r"\1\n\n[TOC]", content)
        content = re.sub(r"\[/TOC\](\S)", r"[/TOC]\n\n\1", content)
        
        # Add spacing after headers (##, ###, etc.) - but not inside tables
        content = re.sub(r"(^#{1,6}\s+[^\n]+)\n([^\n#])", r"\1\n\n\2", content, flags=re.MULTILINE)
        
        # Add spacing before headers (but not if already spaced)
        content = re.sub(r"([^\n])\n(^#{1,6}\s+)", r"\1\n\n\2", content, flags=re.MULTILINE)
        
        # Normalize excessive blank lines (max 2 consecutive) - but preserve table placeholders
        content = re.sub(r"\n{4,}", "\n\n\n", content)
        content = re.sub(r"\n\s*\n\s*\n+", "\n\n", content)
        
        # Ensure proper spacing around list items
        content = re.sub(r"([^\n])\n(^\s*[-*+]\s)", r"\1\n\n\2", content, flags=re.MULTILINE)
        content = re.sub(r"(^\s*[-*+]\s[^\n]+)\n([^-\n\s*+])", r"\1\n\n\2", content, flags=re.MULTILINE)
        
        # Clean up any remaining artifacts
        content = content.strip()
        
        # Ensure tags are properly formatted (no extra spaces inside)
        content = re.sub(r"\[\s*(IMAGE|TOC|HEADER)\s*\]", r"[\1]", content)
        content = re.sub(r"\[\s*/(IMAGE|TOC|HEADER)\s*\]", r"[/\1]", content)
        
        # Final cleanup: ensure consistent spacing (but not in tables)
        # Remove spaces before newlines
        content = re.sub(r" +\n", "\n", content)
        # Normalize final spacing
        content = re.sub(r"\n{3,}", "\n\n", content)
        
        # Step 2: Process and format tables, then re-insert them
        def format_table_content(table_content: str) -> str:
            """Format a single table with proper column alignment and cell sanitization."""
            lines = [line.rstrip() for line in table_content.strip().split('\n')]
            table_rows = []
            max_cols = 0
            separator_added = False
            
            # First pass: parse table rows and find max columns
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                
                if line.startswith('|') and line.endswith('|'):
                    # Parse cells
                    cells = [sanitize_cell(c.strip()) for c in line.split('|')[1:-1]]
                    num_cols = len(cells)
                    max_cols = max(max_cols, num_cols)
                    
                    # Check if separator row
                    is_separator = all(c in ['-', ':', ' '] or not c for c in cells)
                    if is_separator:
                        if not separator_added:
                            table_rows.append((None, True))  # Will be regenerated
                            separator_added = True
                    else:
                        table_rows.append((cells, False))
            
            if max_cols == 0:
                return table_content  # Return original if invalid
            
            # Second pass: rebuild table with consistent columns
            formatted_lines = []
            for cells, is_separator in table_rows:
                if is_separator:
                    formatted_lines.append("| " + " | ".join(["---"] * max_cols) + " |")
                else:
                    # Pad or truncate to max_cols
                    while len(cells) < max_cols:
                        cells.append("")
                    cells = cells[:max_cols]
                    formatted_lines.append("| " + " | ".join(cells) + " |")
            
            # Ensure we have a separator if we have rows
            if formatted_lines and not separator_added:
                # Insert separator after first row
                formatted_lines.insert(1, "| " + " | ".join(["---"] * max_cols) + " |")
            
            return "\n".join(formatted_lines)
        
        # Process all extracted tables
        processed_tables = {}
        for placeholder, table_content in table_placeholders.items():
            processed_tables[placeholder] = format_table_content(table_content)
        
        # Step 3: Re-insert processed tables with proper spacing
        for placeholder, processed_table in processed_tables.items():
            # Add spacing around table
            replacement = f"\n\n{processed_table}\n\n"
            content = content.replace(placeholder, replacement)
        
        # Final cleanup: remove excessive blank lines around tables
        content = re.sub(r"\n{4,}", "\n\n\n", content)

        return content


