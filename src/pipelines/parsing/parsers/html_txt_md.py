"""HTML/TXT/MD parser for ParseForge."""

import logging
from pathlib import Path

from bs4 import BeautifulSoup

from src.schema.document import BlockType, Document, Page, TextBlock, TitleBlock
from src.utils.exceptions import ParserError

logger = logging.getLogger(__name__)


def parse_html(file_path: str) -> Document:
    """Parse HTML file."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        soup = BeautifulSoup(content, "html.parser")
        blocks = []

        # Extract text from common elements
        for element in soup.find_all(["p", "h1", "h2", "h3", "h4", "h5", "h6", "div"]):
            text = element.get_text().strip()
            if not text:
                continue

            tag = element.name
            if tag.startswith("h"):
                level = int(tag[1]) if len(tag) > 1 and tag[1].isdigit() else 1
                block = TitleBlock(
                    block_type=BlockType.TITLE,
                    text=text,
                    page_index=0,
                    level=level,
                )
            else:
                block = TextBlock(
                    block_type=BlockType.TEXT,
                    text=text,
                    page_index=0,
                )
            blocks.append(block)

        page = Page(page_index=0, width=612, height=792, blocks=blocks)
        document = Document(
            file_path=file_path,
            file_name=Path(file_path).name,
            pages=[page],
            total_pages=1,
        )

        return document

    except Exception as e:
        raise ParserError(f"Failed to parse HTML: {e}") from e


def parse_txt_md(file_path: str) -> Document:
    """Parse TXT or MD file."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            lines = f.readlines()

        blocks = []
        current_text = []

        for line in lines:
            line = line.rstrip()
            if line.startswith("#"):
                # Heading
                level = len(line) - len(line.lstrip("#"))
                text = line.lstrip("# ").strip()
                if text:
                    blocks.append(
                        TitleBlock(
                            block_type=BlockType.TITLE,
                            text=text,
                            page_index=0,
                            level=level,
                        )
                    )
            elif line.strip():
                current_text.append(line)
            else:
                if current_text:
                    blocks.append(
                        TextBlock(
                            block_type=BlockType.TEXT,
                            text="\n".join(current_text),
                            page_index=0,
                        )
                    )
                    current_text = []

        if current_text:
            blocks.append(
                TextBlock(
                    block_type=BlockType.TEXT,
                    text="\n".join(current_text),
                    page_index=0,
                )
            )

        page = Page(page_index=0, width=612, height=792, blocks=blocks)
        document = Document(
            file_path=file_path,
            file_name=Path(file_path).name,
            pages=[page],
            total_pages=1,
        )

        return document

    except Exception as e:
        raise ParserError(f"Failed to parse TXT/MD: {e}") from e

