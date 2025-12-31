"""Cross-page table merging."""

import logging
from typing import List, Optional, Tuple

from bs4 import BeautifulSoup

from src.schema.document import TableBlock

logger = logging.getLogger(__name__)

CONTINUATION_MARKERS = ["(续)", "(续表)", "(continued)", "(cont.)"]


def detect_table_headers(
    soup1: BeautifulSoup, soup2: BeautifulSoup, max_header_rows: int = 5
) -> Tuple[int, bool, List[str]]:
    """
    Detect and compare table headers.

    Args:
        soup1: First table BeautifulSoup
        soup2: Second table BeautifulSoup
        max_header_rows: Maximum header rows to check

    Returns:
        Tuple of (header_count, headers_match, header_texts)
    """
    rows1 = soup1.find_all("tr")
    rows2 = soup2.find_all("tr")

    min_rows = min(len(rows1), len(rows2), max_header_rows)
    header_rows = 0
    headers_match = True
    header_texts = []

    for i in range(min_rows):
        cells1 = rows1[i].find_all(["td", "th"])
        cells2 = rows2[i].find_all(["td", "th"])

        if len(cells1) != len(cells2):
            break

        row_texts1 = [cell.get_text().strip() for cell in cells1]
        row_texts2 = [cell.get_text().strip() for cell in cells2]

        if row_texts1 == row_texts2:
            header_rows += 1
            header_texts.append(row_texts1)
        else:
            break

    return header_rows, headers_match, header_texts


def can_merge_tables(
    table1: TableBlock, table2: TableBlock
) -> Tuple[bool, Optional[BeautifulSoup], Optional[BeautifulSoup], str, str]:
    """
    Check if two tables can be merged.

    Args:
        table1: First table block
        table2: Second table block

    Returns:
        Tuple of (can_merge, soup1, soup2, html1, html2)
    """
    if table1.html is None or table2.html is None:
        return False, None, None, "", ""

    try:
        soup1 = BeautifulSoup(table1.html, "html.parser")
        soup2 = BeautifulSoup(table2.html, "html.parser")

        # Check for continuation markers
        text1 = soup1.get_text().lower()
        text2 = soup2.get_text().lower()

        has_continuation = any(marker.lower() in text2 for marker in CONTINUATION_MARKERS)

        # Check header match
        header_count, headers_match, _ = detect_table_headers(soup1, soup2)

        can_merge = has_continuation or (headers_match and header_count > 0)

        return can_merge, soup1, soup2, table1.html, table2.html

    except Exception as e:
        logger.warning(f"Error checking table merge: {e}")
        return False, None, None, "", ""


def merge_tables(table1: TableBlock, table2: TableBlock) -> Optional[str]:
    """
    Merge two tables into one HTML string.

    Args:
        table1: First table block
        table2: Second table block

    Returns:
        Merged HTML table string, or None if merge failed
    """
    can_merge, soup1, soup2, _, _ = can_merge_tables(table1, table2)

    if not can_merge or soup1 is None or soup2 is None:
        return None

    try:
        # Detect headers
        header_count, _, _ = detect_table_headers(soup1, soup2)

        # Get rows
        rows1 = soup1.find_all("tr")
        rows2 = soup2.find_all("tr")

        # Find tbody
        tbody1 = soup1.find("tbody") or soup1.find("table")

        if tbody1 and len(rows2) > header_count:
            tbody2 = soup2.find("tbody") or soup2.find("table")
            if tbody2:
                # Append rows from table2 (skip headers)
                for row in rows2[header_count:]:
                    row_copy = BeautifulSoup(str(row), "html.parser").find("tr")
                    if row_copy:
                        tbody1.append(row_copy)

        return str(soup1)

    except Exception as e:
        logger.error(f"Error merging tables: {e}")
        return None


def merge_cross_page_tables(tables: List[TableBlock]) -> List[TableBlock]:
    """
    Merge tables across pages.

    Args:
        tables: List of TableBlocks from all pages

    Returns:
        List of merged TableBlocks
    """
    if len(tables) <= 1:
        return tables

    merged = []
    i = 0

    while i < len(tables):
        current_table = tables[i]
        merged.append(current_table)

        # Check if next table can be merged with current
        if i + 1 < len(tables):
            next_table = tables[i + 1]
            merged_html = merge_tables(current_table, next_table)

            if merged_html:
                # Update current table with merged HTML
                merged[-1] = current_table.model_copy(update={"html": merged_html})
                # Skip next table as it's been merged
                i += 2
                continue

        i += 1

    return merged

