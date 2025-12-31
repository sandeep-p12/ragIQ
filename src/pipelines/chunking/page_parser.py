"""Page-aware processing with safe cross-page continuation merging."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Optional, Tuple

from src.pipelines.chunking.repair import RepairRecord, RepairResult
from src.schema.chunk import PageBlock


@dataclass
class PageBlock:
    """Represents a page block with metadata."""
    page_no: int
    content: str  # Repaired content
    raw_lines: List[str]
    start_line: int
    end_line: int
    structure_confidence: float
    repair_applied: List[RepairRecord]
    page_span: Tuple[int, int] = None  # Will be set during merging
    page_nos: List[int] = None  # Will be set during merging
    
    def __post_init__(self):
        """Initialize page_span and page_nos after creation."""
        if self.page_span is None:
            self.page_span = (self.page_no, self.page_no)
        if self.page_nos is None:
            self.page_nos = [self.page_no]


def detect_page_markers(text: str) -> List[Tuple[int, int, str]]:
    """Detect page markers in text.
    
    Finds normalized page markers and returns (line_no, page_no, marker_text).
    
    Args:
        text: Input text
    
    Returns:
        List[Tuple[int, int, str]]: List of (line_number, page_number, marker_text)
    """
    markers = []
    lines = text.split("\n")
    
    # Pattern for normalized page markers: --- Page N ---
    pattern = r'---\s*Page\s+(\d+)\s*---'
    
    for i, line in enumerate(lines):
        match = re.match(pattern, line.strip(), re.IGNORECASE)
        if match:
            page_no = int(match.group(1))
            markers.append((i, page_no, line.strip()))
    
    return markers


def split_into_page_blocks(text: str, repair_result: RepairResult) -> List[PageBlock]:
    """Split text into PageBlocks with repair metadata.
    
    Args:
        text: Repaired markdown text
        repair_result: Repair result with metadata
    
    Returns:
        List[PageBlock]: List of page blocks
    """
    markers = detect_page_markers(text)
    lines = text.split("\n")
    blocks = []
    
    if not markers:
        # No page markers - treat entire document as single page
        blocks.append(PageBlock(
            page_no=1,
            content=text,
            raw_lines=lines,
            start_line=0,
            end_line=len(lines) - 1,
            structure_confidence=repair_result.structure_confidence,
            repair_applied=[]
        ))
        return blocks
    
    # Extract repair records by line range
    all_repair_records = []
    for repair_type, records in repair_result.repair_applied.items():
        all_repair_records.extend(records)
    
    # Create blocks: content after each marker belongs to that page
    for i, (line_no, page_no, marker_text) in enumerate(markers):
        # Content starts after the marker line
        start_line = line_no + 1
        
        # Content ends before the next marker (or end of file)
        if i < len(markers) - 1:
            # There's a next marker
            next_line_no = markers[i + 1][0]
            end_line = next_line_no - 1
        else:
            # This is the last marker, content goes to end of file
            end_line = len(lines) - 1
        
        # Extract content for this block
        if end_line >= start_line:
            block_lines = lines[start_line:end_line + 1]
            block_content = "\n".join(block_lines)
        else:
            block_lines = []
            block_content = ""
        
        # Get repair records for this block
        block_repairs = [
            r for r in all_repair_records
            if start_line <= r.location[0] <= end_line
        ]
        
        blocks.append(PageBlock(
            page_no=page_no,
            content=block_content,
            raw_lines=block_lines,
            start_line=start_line,
            end_line=end_line,
            structure_confidence=repair_result.structure_confidence,
            repair_applied=block_repairs
        ))
    
    return blocks


def detect_continuation(
    page_block: PageBlock, 
    next_block: PageBlock, 
    aggressiveness: str = "medium"
) -> bool:
    """Detect if next block is a continuation of current block.
    
    Conservative heuristics:
    - Paragraph: ends mid-sentence or no blank line before marker
    - List: ends in list context, next begins with same pattern
    - Table: ends with table row, next begins with row-like pipes
    - Image: [IMAGE] not closed or sequential image blocks
    
    Args:
        page_block: Current page block
        next_block: Next page block
        aggressiveness: "low", "medium", or "high"
    
    Returns:
        bool: True if continuation detected
    """
    if not page_block.content.strip() or not next_block.content.strip():
        return False
    
    current_lines = page_block.raw_lines
    next_lines = next_block.raw_lines
    
    if not current_lines or not next_lines:
        return False
    
    last_line = current_lines[-1].strip()
    first_line = next_lines[0].strip()
    
    # 1. Paragraph continuation
    # Ends mid-sentence (no terminal punctuation) or no blank line before marker
    if last_line and not re.match(r'.*[.!?]\s*$', last_line):
        # Check if next line looks like continuation
        if first_line and not first_line.startswith(('#', '-', '*', '|', '[')):
            if aggressiveness in ["medium", "high"]:
                return True
    
    # 2. List continuation
    # Current ends in list context, next begins with same pattern
    last_is_list = bool(re.match(r'^(\s*)([-*+]|\d+\.)\s+', last_line))
    first_is_list = bool(re.match(r'^(\s*)([-*+]|\d+\.)\s+', first_line))
    
    if last_is_list and first_is_list:
        # Check if same list pattern
        last_match = re.match(r'^(\s*)([-*+]|\d+\.)\s+', last_line)
        first_match = re.match(r'^(\s*)([-*+]|\d+\.)\s+', first_line)
        
        if last_match and first_match:
            last_indent = len(last_match.group(1))
            first_indent = len(first_match.group(1))
            
            # Same or compatible indentation
            if abs(last_indent - first_indent) <= 2:
                return True
    
    # 3. Table continuation
    # Current ends with table row, next begins with row-like pipes
    last_is_table = '|' in last_line and len(last_line.split('|')) >= 2
    first_is_table = '|' in first_line and len(first_line.split('|')) >= 2
    
    if last_is_table and first_is_table:
        # Check if not a header separator
        is_separator = bool(re.match(r'^[\s\-\|:]+$', first_line))
        if not is_separator:
            # Same number of columns (rough check)
            last_cols = len(last_line.split('|'))
            first_cols = len(first_line.split('|'))
            if abs(last_cols - first_cols) <= 1:
                return True
    
    # 4. Image continuation
    # [IMAGE] not closed or sequential image blocks
    current_content = page_block.content
    next_content = next_block.content
    
    # Check if [IMAGE] block not closed
    open_images = current_content.count('[IMAGE]')
    close_images = current_content.count('[/IMAGE]')
    
    if open_images > close_images:
        return True
    
    # Check if sequential image blocks
    if '[IMAGE]' in current_content and '[IMAGE]' in next_content:
        if aggressiveness == "high":
            return True
    
    return False


def merge_continuations(
    blocks: List[PageBlock], 
    config
) -> List[PageBlock]:
    """Merge page blocks when continuation is detected.
    
    Only merges when high-confidence heuristics trigger.
    Always preserves page_span and page_nos.
    
    Args:
        blocks: List of page blocks
        config: ChunkConfig with merge settings
    
    Returns:
        List[PageBlock]: Merged blocks
    """
    if not config.enable_cross_page_merge or len(blocks) <= 1:
        return blocks
    
    merged = []
    i = 0
    
    while i < len(blocks):
        current_block = blocks[i]
        
        # Check if we can merge with next block
        if i < len(blocks) - 1:
            next_block = blocks[i + 1]
            
            if detect_continuation(current_block, next_block, config.cross_page_merge_aggressiveness):
                # Merge blocks
                merged_content = current_block.content + "\n" + next_block.content
                merged_lines = current_block.raw_lines + next_block.raw_lines
                
                # Combine repair records
                merged_repairs = current_block.repair_applied + next_block.repair_applied
                
                # Compute combined structure confidence (average)
                merged_confidence = (
                    current_block.structure_confidence + next_block.structure_confidence
                ) / 2
                
                # Create merged block
                merged_block = PageBlock(
                    page_no=current_block.page_no,  # Use first page number
                    content=merged_content,
                    raw_lines=merged_lines,
                    start_line=current_block.start_line,
                    end_line=next_block.end_line,
                    structure_confidence=merged_confidence,
                    repair_applied=merged_repairs,
                    page_span=(current_block.page_no, next_block.page_no),
                    page_nos=list(range(current_block.page_no, next_block.page_no + 1))
                )
                
                merged.append(merged_block)
                i += 2  # Skip next block as it's merged
            else:
                merged.append(current_block)
                i += 1
        else:
            merged.append(current_block)
            i += 1
    
    return merged

