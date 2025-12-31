"""CSV parser for ParseForge."""

import logging
from pathlib import Path

import pandas as pd

from src.schema.document import BlockType, Document, Page, TableBlock
from src.utils.exceptions import ParserError

logger = logging.getLogger(__name__)


def parse_csv(file_path: str) -> Document:
    """
    Parse CSV file.

    Args:
        file_path: Path to CSV file

    Returns:
        Document object
    """
    try:
        df = pd.read_csv(file_path)

        # Convert to cells (ensure all values are strings)
        cells = [[str(val) for val in df.columns.tolist()]] + [[str(val) for val in row] for row in df.values.tolist()]

        # Create table block
        table = TableBlock(
            block_type=BlockType.TABLE,
            page_index=0,
            cells=cells,
            num_rows=len(cells),
            num_cols=len(cells[0]) if cells else 0,
        )

        # Create page
        page = Page(page_index=0, width=612, height=792, blocks=[table])

        # Create document
        document = Document(
            file_path=file_path,
            file_name=Path(file_path).name,
            pages=[page],
            total_pages=1,
        )

        return document

    except Exception as e:
        raise ParserError(f"Failed to parse CSV: {e}") from e

