"""Core ParseForge orchestrator."""

import logging
from pathlib import Path
from typing import Callable, Dict, List, Optional

from src.config.parsing import ParseForgeConfig
from src.config.parsing_strategies import StrategyEnum
from src.formatters.image import ImageVisionLLMFormatter
from src.formatters.markdown import blocks_to_markdown
from src.formatters.table import TableLLMFormatter
from src.pipelines.parsing.parsers.csv import parse_csv
from src.pipelines.parsing.parsers.docx import parse_docx
from src.pipelines.parsing.parsers.html_txt_md import parse_html, parse_txt_md
from src.pipelines.parsing.parsers.pdf import PDFParser
from src.pipelines.parsing.parsers.pptx import parse_pptx
from src.pipelines.parsing.parsers.xlsx import parse_xlsx
from src.providers.layout.yolo import YOLOLayoutDetector
from src.pipelines.parsing.processing.reading_order import sort_blocks_by_reading_order
from src.pipelines.parsing.processing.para_split import split_paragraphs
from src.pipelines.parsing.processing.magic import MagicModel
from src.pipelines.parsing.processing.table_extractor import extract_table
from src.pipelines.parsing.processing.table_merger import merge_cross_page_tables
from src.schema.document import Document, ImageBlock
from src.utils.checkpoint import Checkpoint
from src.utils.exceptions import ParseForgeException

logger = logging.getLogger(__name__)


class ParseForge:
    """Main ParseForge orchestrator class."""

    def __init__(
        self,
        config: Optional[ParseForgeConfig] = None,
        progress_callback: Optional[Callable[[str, float, Optional[Dict]], None]] = None,
    ):
        """
        Initialize ParseForge.

        Args:
            config: ParseForge configuration
            progress_callback: Optional callback for progress updates (stage, progress, output_data)
        """
        self.config = config or ParseForgeConfig()
        self.progress_callback = progress_callback

        # Initialize components (lazy loading for models that may not be available)
        try:
            self.layout_detector = YOLOLayoutDetector(
                device=self.config.device,
                config=self.config,
            )
        except Exception as e:
            logger.warning(f"Failed to initialize layout detector: {e}. Layout detection will be disabled.")
            self.layout_detector = None
        
        self.table_formatter = TableLLMFormatter(config=self.config)
        self.image_formatter = ImageVisionLLMFormatter(config=self.config)
        self.checkpoint = Checkpoint(self.config.checkpoint_dir)

    def _update_progress(self, stage: str, progress: float, output_data: Optional[Dict] = None):
        """Update progress via callback."""
        if self.progress_callback:
            self.progress_callback(stage, progress, output_data)

    def parse(
        self,
        file_path: str,
        strategy: StrategyEnum = StrategyEnum.AUTO,
        start_page: Optional[int] = None,
        end_page: Optional[int] = None,
    ) -> Document:
        """
        Parse a document.

        Args:
            file_path: Path to document file
            strategy: Parsing strategy
            start_page: Start page (0-based)
            end_page: End page (0-based, None for all)

        Returns:
            Parsed Document
        """
        file_path_obj = Path(file_path)
        suffix = file_path_obj.suffix.lower()

        # File loading output
        file_size = file_path_obj.stat().st_size if file_path_obj.exists() else 0
        self._update_progress(
            "file_loading",
            0.0,
            {
                "file_name": file_path_obj.name,
                "file_size": file_size,
                "file_type": suffix,
            },
        )

        # Route to appropriate parser
        if suffix == ".pdf":
            parser = PDFParser(config=self.config, progress_callback=self._update_progress)
            # Convert None to proper defaults for PDFParser
            pdf_start = start_page if start_page is not None else 0
            document = parser.parse(file_path, strategy, pdf_start, end_page)
        elif suffix == ".docx":
            document = parse_docx(file_path)
        elif suffix == ".pptx":
            document = parse_pptx(file_path)
        elif suffix == ".xlsx" or suffix == ".xls":
            document = parse_xlsx(file_path)
        elif suffix == ".csv":
            document = parse_csv(file_path)
        elif suffix in [".html", ".htm"]:
            document = parse_html(file_path)
        elif suffix in [".txt", ".md"]:
            document = parse_txt_md(file_path)
        else:
            raise ParseForgeException(f"Unsupported file type: {suffix}")

        self._update_progress("Parsing complete", 1.0)

        return document

    def _format_tables_in_markdown(self, markdown: str) -> str:
        """
        Format and fix tables in markdown content.
        Ensures proper column alignment, cell sanitization, and table structure.
        
        Args:
            markdown: Markdown string that may contain tables
            
        Returns:
            Markdown string with properly formatted tables
        """
        import re
        
        def sanitize_cell(cell: str) -> str:
            """Sanitize cell content: remove pipes, normalize whitespace."""
            if not cell:
                return ""
            # Replace pipe characters with space (pipes break markdown tables)
            cell = cell.replace('|', ' ')
            # Normalize whitespace
            cell = re.sub(r'\s+', ' ', cell.strip())
            return cell
        
        def format_table_content(table_content: str) -> str:
            """Format a single table with proper column alignment and cell sanitization."""
            lines = [line.rstrip() for line in table_content.strip().split('\n')]
            table_rows = []
            max_cols = 0
            separator_added = False
            separator_position = -1
            
            # First pass: parse table rows and find max columns
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                
                if line.startswith('|') and line.endswith('|'):
                    # Parse cells - handle empty cells properly
                    raw_cells = line.split('|')
                    # Remove first and last empty elements
                    if raw_cells and not raw_cells[0].strip():
                        raw_cells = raw_cells[1:]
                    if raw_cells and not raw_cells[-1].strip():
                        raw_cells = raw_cells[:-1]
                    
                    cells = [sanitize_cell(c.strip()) for c in raw_cells]
                    num_cols = len(cells)
                    max_cols = max(max_cols, num_cols)
                    
                    # Check if separator row (contains dashes, colons, or is mostly empty)
                    cell_chars = ''.join(cells).replace(' ', '').replace('-', '').replace(':', '')
                    is_separator = len(cell_chars) == 0 or all(c in ['-', ':', ' '] or not c for c in cells)
                    
                    if is_separator:
                        if not separator_added:
                            separator_position = len(table_rows)
                            table_rows.append((None, True))  # Will be regenerated
                            separator_added = True
                    else:
                        table_rows.append((cells, False))
            
            if max_cols == 0:
                return table_content  # Return original if invalid
            
            # Second pass: rebuild table with consistent columns
            formatted_lines = []
            for idx, (cells, is_separator) in enumerate(table_rows):
                if is_separator:
                    # Insert separator at correct position (after first data row)
                    if idx == 0:
                        # Separator is first - move it after first data row
                        continue
                    else:
                        formatted_lines.append("| " + " | ".join(["---"] * max_cols) + " |")
                else:
                    # Pad or truncate to max_cols
                    while len(cells) < max_cols:
                        cells.append("")
                    cells = cells[:max_cols]
                    formatted_lines.append("| " + " | ".join(cells) + " |")
                    
                    # Insert separator after first data row if not already present
                    if len(formatted_lines) == 1 and not separator_added:
                        formatted_lines.append("| " + " | ".join(["---"] * max_cols) + " |")
                        separator_added = True
            
            # If separator was first and we have data rows, add it after first row
            if separator_position == 0 and formatted_lines:
                formatted_lines.insert(1, "| " + " | ".join(["---"] * max_cols) + " |")
            
            # Ensure we have a separator if we have rows but no separator yet
            if formatted_lines and not separator_added and len(formatted_lines) > 0:
                # Insert separator after first row
                formatted_lines.insert(1, "| " + " | ".join(["---"] * max_cols) + " |")
            
            return "\n".join(formatted_lines)
        
        # Extract and protect tables from any modifications
        table_placeholders = {}
        table_counter = 0
        
        def extract_table(match):
            nonlocal table_counter
            table_content = match.group(1)
            placeholder = f"__TABLE_PLACEHOLDER_{table_counter}__"
            table_placeholders[placeholder] = table_content
            table_counter += 1
            return placeholder
        
        # Extract tables with [TABLE] tags
        table_pattern = r"\[TABLE\](.*?)\[/TABLE\]"
        markdown = re.sub(table_pattern, extract_table, markdown, flags=re.DOTALL)
        
        # Extract raw markdown tables (consecutive lines starting and ending with |)
        # First, find potential table blocks (consecutive lines with |)
        def extract_raw_table(match):
            nonlocal table_counter
            table_content = match.group(0)
            # Verify it looks like a table
            lines = [l.strip() for l in table_content.strip().split('\n') if l.strip()]
            if len(lines) < 2:
                return match.group(0)  # Need at least 2 rows
            
            # Check if it has table-like structure (all lines start and end with |)
            all_table_lines = all(line.startswith('|') and line.endswith('|') for line in lines)
            if not all_table_lines:
                return match.group(0)
            
            # Check for separator row (contains --- or is all dashes/spaces)
            has_separator = False
            for line in lines:
                # Check if line is a separator (contains --- or is mostly dashes/spaces/colons)
                cell_content = line.replace('|', '').strip()
                if '---' in line or re.match(r'^[\s\-:]+$', cell_content):
                    has_separator = True
                    break
            
            # Accept as table if it has separator OR if it has at least 3 rows (likely a table)
            if has_separator or len(lines) >= 3:
                placeholder = f"__TABLE_PLACEHOLDER_{table_counter}__"
                table_placeholders[placeholder] = table_content
                table_counter += 1
                return placeholder
            return match.group(0)  # Return original if not a valid table
        
        # Pattern: consecutive lines that start and end with |
        # Match 2+ consecutive lines that look like table rows
        raw_table_pattern = r"(?:^\s*\|[^\n]*\|\s*(?:\r?\n|$)){2,}"
        markdown = re.sub(raw_table_pattern, extract_raw_table, markdown, flags=re.MULTILINE)
        
        # Process all extracted tables
        processed_tables = {}
        for placeholder, table_content in table_placeholders.items():
            processed_tables[placeholder] = format_table_content(table_content)
        
        # Re-insert processed tables
        for placeholder, processed_table in processed_tables.items():
            # Add spacing around table
            replacement = f"\n\n{processed_table}\n\n"
            markdown = markdown.replace(placeholder, replacement)
        
        # Final cleanup: remove excessive blank lines around tables
        markdown = re.sub(r"\n{4,}", "\n\n\n", markdown)
        
        return markdown

    def to_markdown(self, document: Document, generate_image_descriptions: bool = True) -> str:
        """
        Convert document to markdown.

        Args:
            document: Document to convert
            generate_image_descriptions: Whether to generate image descriptions using vision LLM

        Returns:
            Markdown string
        """
        from src.schema.document import TableBlock
        
        all_blocks = document.get_all_blocks()

        # Format tables with LLM if enabled
        if self.config.llm_provider != "none" and self.table_formatter is not None and hasattr(self.table_formatter, 'llm_provider') and self.table_formatter.llm_provider is not None:
            self._update_progress("Formatting tables with LLM", 0.0)
            table_blocks = [b for b in all_blocks if isinstance(b, TableBlock)]
            previous_table_md = None
            
            for i, table_block in enumerate(table_blocks):
                try:
                    formatted_table = self.table_formatter.format_table(table_block, previous_table_md)
                    # Update the table block in the document
                    table_idx = all_blocks.index(table_block)
                    all_blocks[table_idx] = formatted_table
                    # Extract markdown for next table context
                    if formatted_table.html and "<!-- MARKDOWN_TABLE_START -->" in formatted_table.html:
                        start_marker = "<!-- MARKDOWN_TABLE_START -->\n"
                        end_marker = "\n<!-- MARKDOWN_TABLE_END -->"
                        start_idx = formatted_table.html.find(start_marker) + len(start_marker)
                        end_idx = formatted_table.html.find(end_marker, start_idx)
                        if end_idx != -1:
                            previous_table_md = formatted_table.html[start_idx:end_idx].strip()
                except Exception as e:
                    logger.warning(f"Failed to format table {i} with LLM: {e}")
            
            self._update_progress("Table formatting complete", 0.3)

        # Generate image descriptions if enabled
        image_descriptions = {}
        if generate_image_descriptions and self.config.llm_provider != "none":
            self._update_progress("Generating image descriptions", 0.3)
            image_blocks = [b for b in all_blocks if isinstance(b, ImageBlock)]
            if image_blocks:
                descriptions = self.image_formatter.describe_images_batch(image_blocks)
                if len(image_blocks) != len(descriptions):
                    raise ValueError(f"Mismatch: {len(image_blocks)} blocks but {len(descriptions)} descriptions")
                image_descriptions = {
                    block.block_id: desc for block, desc in zip(image_blocks, descriptions)
                }
            self._update_progress("Image descriptions complete", 0.7)

        # Convert to markdown
        # Check for pages with vision LLM markdown (processed with images)
        # These pages already have complete markdown with layout maintained
        self._update_progress("Generating markdown", 0.7)
        
        markdown_parts = []
        for page in document.pages:
            # Check if page has LLM_FULL markdown (complete document parsed with LLM)
            if page.metadata.get('llm_full_processed') and page.metadata.get('llm_full_markdown'):
                # Use the LLM-generated markdown, but ensure tables are properly formatted
                page_markdown = page.metadata['llm_full_markdown']
                # Apply final table formatting to ensure all tables are properly formatted
                page_markdown = self._format_tables_in_markdown(page_markdown)
                markdown_parts.append(page_markdown)
                logger.debug(f"Using LLM_FULL markdown for page {page.page_index}")
            # Check if page has vision LLM markdown (processed with images)
            elif page.metadata.get('vision_llm_processed') and page.metadata.get('vision_llm_markdown'):
                # Use the LLM-generated markdown, but ensure tables are properly formatted
                page_markdown = page.metadata['vision_llm_markdown']
                # Apply final table formatting to ensure all tables are properly formatted
                page_markdown = self._format_tables_in_markdown(page_markdown)
                markdown_parts.append(page_markdown)
                logger.debug(f"Using vision LLM markdown for page {page.page_index}")
            else:
                # Process blocks normally for pages without LLM processing
                page_blocks = page.blocks
                page_markdown = blocks_to_markdown(
                    page_blocks,
                    image_formatter=self.image_formatter if generate_image_descriptions else None,
                    image_descriptions=image_descriptions if image_descriptions else None,
                )
                markdown_parts.append(page_markdown)
        
        # Combine all page markdowns, maintaining page order
        # Add page number identifiers for all pages and all strategies
        final_markdown = []
        
        for i, page_md in enumerate(markdown_parts):
            if page_md.strip():  # Only add non-empty pages
                # Add page identifier for all pages (including first page)
                # Use actual page index for all strategies
                page_number = document.pages[i].page_index + 1 if i < len(document.pages) else i + 1
                page_identifier = f"--- Page {page_number} ---"
                
                final_markdown.append(f"\n\n{page_identifier}\n\n")
                final_markdown.append(page_md)
        
        markdown = "".join(final_markdown)
        self._update_progress("Markdown generation complete", 1.0)

        return markdown

