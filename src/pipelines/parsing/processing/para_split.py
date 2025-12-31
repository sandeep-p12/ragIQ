"""Paragraph splitting and merging."""

import logging
import re
from typing import List

from src.schema.document import Block, BlockType, TextBlock, TitleBlock

logger = logging.getLogger(__name__)

LIST_END_FLAG = [".", ":", ";", "。", "：", "；"]


def is_list_block(block: TextBlock) -> bool:
    """
    Determine if a block is a list.

    Args:
        block: Text block to check

    Returns:
        True if block is a list
    """
    if not block.text:
        return False

    lines = block.text.split("\n")
    if len(lines) < 2:
        return False

    # Check for numbered or bulleted items
    numbered_count = sum(1 for line in lines if re.match(r"^\s*[\d\.\)]\s+", line.strip()))
    bulleted_count = sum(1 for line in lines if re.match(r"^\s*[-•*]\s+", line.strip()))

    return (numbered_count + bulleted_count) / len(lines) > 0.5


def is_index_block(block: TextBlock) -> bool:
    """
    Determine if a block is an index/TOC.

    Args:
        block: Text block to check

    Returns:
        True if block is an index
    """
    if not block.text:
        return False

    lines = block.text.split("\n")
    if len(lines) < 2:
        return False

    # Check for index patterns (numbers at start or end)
    numbered_pattern = re.compile(r"^\s*\d+[\.\)]\s+.*\s+\d+$")
    numbered_count = sum(1 for line in lines if numbered_pattern.match(line.strip()))

    return numbered_count / len(lines) > 0.6


def merge_text_blocks(block1: TextBlock, block2: TextBlock) -> TextBlock:
    """
    Merge two text blocks if appropriate.

    Args:
        block1: First text block
        block2: Second text block

    Returns:
        Merged text block
    """
    # Don't merge if either is a list or index
    if is_list_block(block1) or is_list_block(block2):
        return block1

    if is_index_block(block1) or is_index_block(block2):
        return block1

    # Merge text
    merged_text = block1.text + " " + block2.text

    # Create merged block
    merged = TextBlock(
        block_type=BlockType.TEXT,
        text=merged_text,
        bbox=block1.bbox,  # Use first block's bbox
        page_index=block1.page_index,
        spans=block1.spans + block2.spans,
    )

    return merged


def split_paragraphs(blocks: List[Block]) -> List[Block]:
    """
    Split and merge paragraphs appropriately.

    Args:
        blocks: List of blocks to process

    Returns:
        Processed blocks with proper paragraph splitting
    """
    result = []
    i = 0

    while i < len(blocks):
        block = blocks[i]

        # Classify block type
        if isinstance(block, TextBlock):
            if is_index_block(block):
                block.block_type = BlockType.INDEX
            elif is_list_block(block):
                block.block_type = BlockType.LIST

            # Try to merge with next block
            if i + 1 < len(blocks) and isinstance(blocks[i + 1], TextBlock):
                next_block = blocks[i + 1]
                if not is_list_block(block) and not is_list_block(next_block):
                    merged = merge_text_blocks(block, next_block)
                    result.append(merged)
                    i += 2
                    continue

        result.append(block)
        i += 1

    return result

