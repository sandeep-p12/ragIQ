"""Text-only LLM formatter for tables."""

import logging
import re
from typing import Optional

from src.config.parsing import ParseForgeConfig
from src.config.prompts import get_table_formatting_prompt
from src.providers.llm.openai_llm import OpenAILLMProvider
from src.schema.document import TableBlock

logger = logging.getLogger(__name__)

CODE_BLOCK_PATTERN = re.compile(r"```(?:markdown|md)?\n(.*?)\n```", re.DOTALL)


class TableLLMFormatter:
    """Format tables using text-only LLM."""

    def __init__(self, config: Optional[ParseForgeConfig] = None):
        """
        Initialize table LLM formatter.

        Args:
            config: ParseForge configuration
        """
        self.config = config or ParseForgeConfig()
        self.llm_provider = OpenAILLMProvider(config=self.config)

    def format_table(self, table: TableBlock, previous_table: Optional[str] = None) -> TableBlock:
        """
        Format table using LLM.

        Args:
            table: Table block to format
            previous_table: Previous table markdown for context

        Returns:
            Formatted table block
        """
        if self.llm_provider.client is None:
            logger.warning("LLM client not available, returning table as-is")
            return table

        # Extract table text
        table_text = self._extract_table_text(table)

        # Create prompt
        prompt = get_table_formatting_prompt(table_text, previous_table)

        try:
            # Call LLM via provider
            content = self.llm_provider.generate(
                prompt=prompt,
                model=self.config.llm_model,
                temperature=0.1,
            )

            # Extract markdown table
            markdown_table = self._extract_markdown_table(content)

            # Store markdown directly for markdown formatter to use
            if markdown_table:
                # Also update HTML for backward compatibility
                table.html = self._markdown_to_html(markdown_table)
                # Store markdown in a way markdown formatter can use
                # We'll use a custom attribute or update cells
                # For now, store in HTML as a comment or update cells structure
                # Actually, let's store it in the table's metadata by updating cells with markdown structure
                # But simpler: store markdown in HTML as data attribute or update cells
                # Best approach: Update the table to have markdown stored
                # Since TableBlock doesn't have markdown field, we'll store it in HTML with a marker
                table.html = f"<!-- MARKDOWN_TABLE_START -->\n{markdown_table}\n<!-- MARKDOWN_TABLE_END -->\n{table.html}"

            return table

        except Exception as e:
            logger.error(f"LLM table formatting failed: {e}")
            return table

    def _extract_table_text(self, table: TableBlock) -> str:
        """Extract text representation of table."""
        if table.cells:
            rows = []
            for row in table.cells:
                rows.append(" | ".join(str(cell) for cell in row))
            return "\n".join(rows)
        return ""


    def _extract_markdown_table(self, content: str) -> str:
        """Extract markdown table from LLM response."""
        # Remove code blocks
        content = CODE_BLOCK_PATTERN.sub(r"\1", content)

        # Find table (lines starting with |)
        lines = content.split("\n")
        table_lines = [line for line in lines if line.strip().startswith("|")]

        return "\n".join(table_lines)

    def _markdown_to_html(self, markdown: str) -> str:
        """Convert markdown table to HTML (simplified)."""
        from bs4 import BeautifulSoup

        lines = markdown.strip().split("\n")
        if not lines:
            return ""

        soup = BeautifulSoup("", "html.parser")
        table = soup.new_tag("table")

        for line in lines:
            if "---" in line:  # Separator
                continue

            cells = [cell.strip() for cell in line.split("|")[1:-1]]  # Remove empty first/last
            tr = soup.new_tag("tr")

            for cell_text in cells:
                td = soup.new_tag("td")
                td.string = cell_text
                tr.append(td)

            table.append(tr)

        return str(table)

