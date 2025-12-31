"""Markdown repair mode: table, list, and section repair with metadata tracking."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Tuple


# RepairRecord and RepairResult are now imported from schema
from src.schema.chunk import RepairRecord, RepairResult


def normalize_page_markers(text: str) -> str:
    """Normalize page markers to standard format.
    
    Supports both `--- Page N ---` and `-- page N --` with case/whitespace tolerance.
    Normalizes to `--- Page N ---` format.
    
    Args:
        text: Input markdown text
    
    Returns:
        str: Text with normalized page markers
    """
    # Combined pattern that matches both formats: --- Page N --- or -- page N --
    # Use a single pattern to avoid double replacement
    # Match 2-4 dashes, then Page, then number, then 2-4 dashes
    pattern = r'-{2,4}\s*[Pp]age\s+(\d+)\s*-{2,4}'
    
    def replace_func(match):
        page_num = match.group(1)
        return f"--- Page {page_num} ---"
    
    # Replace with single pattern
    text = re.sub(pattern, replace_func, text)
    
    return text


def repair_tables(lines: List[str]) -> Tuple[List[str], List[RepairRecord]]:
    """Repair malformed tables.
    
    - Detect table-like blocks (pipe density + alignment row patterns)
    - If header separator missing: infer header = first row, synthesize separator
    - If too broken: preserve as TableCandidate, chunk by row-like lines
    
    Args:
        lines: List of lines to repair
    
    Returns:
        Tuple[List[str], List[RepairRecord]]: Repaired lines and repair records
    """
    repaired_lines = lines.copy()
    repair_records = []
    
    i = 0
    while i < len(repaired_lines):
        line = repaired_lines[i].strip()
        
        # Detect potential table start (has pipes)
        if '|' in line and len(line.split('|')) >= 2:
            table_start = i
            table_lines = [line]
            i += 1
            
            # Collect consecutive table-like lines
            while i < len(repaired_lines):
                next_line = repaired_lines[i].strip()
                if '|' in next_line and len(next_line.split('|')) >= 2:
                    table_lines.append(next_line)
                    i += 1
                elif re.match(r'^[\s\-\|:]+$', next_line):  # Alignment separator row
                    table_lines.append(next_line)
                    i += 1
                else:
                    break
            
            # Check if we have a table
            if len(table_lines) >= 2:
                # Check if alignment separator exists
                has_separator = any(re.match(r'^[\s\-\|:]+$', line) for line in table_lines)
                
                if not has_separator and len(table_lines) >= 2:
                    # Missing separator - infer header and add separator
                    header_line = table_lines[0]
                    num_cols = len(header_line.split('|')) - 1
                    
                    # Create separator: | --- | --- | ... |
                    separator = '|' + '|'.join([' --- '] * num_cols) + '|'
                    
                    # Insert separator after header
                    repaired_lines.insert(table_start + 1, separator)
                    
                    repair_records.append(RepairRecord(
                        repair_type="table_repair",
                        location=(table_start, table_start + len(table_lines)),
                        reason="Missing table header separator",
                        original="\n".join(table_lines),
                        repaired=table_lines[0] + "\n" + separator + "\n" + "\n".join(table_lines[1:])
                    ))
                    
                    i += 1  # Adjust for inserted line
                elif len(table_lines) == 1:
                    # Single line with pipes - might be too broken
                    # Mark as TableCandidate (will be handled in chunking)
                    repair_records.append(RepairRecord(
                        repair_type="table_repair",
                        location=(table_start, table_start),
                        reason="Single-line table-like content, treating as TableCandidate",
                        original=table_lines[0],
                        repaired=table_lines[0]
                    ))
        else:
            i += 1
    
    return repaired_lines, repair_records


def repair_lists(lines: List[str]) -> Tuple[List[str], List[RepairRecord]]:
    """Repair malformed lists.
    
    - Detect list blocks by indentation/prefix patterns (even if markers vary)
    - Treat each inferred item as atomic; never split an item
    
    Args:
        lines: List of lines to repair
    
    Returns:
        Tuple[List[str], List[RepairRecord]]: Repaired lines and repair records
    """
    repaired_lines = lines.copy()
    repair_records = []
    
    i = 0
    while i < len(repaired_lines):
        line = repaired_lines[i]
        stripped = line.strip()
        
        # Detect list item patterns: -, *, +, or numbered
        list_pattern = re.match(r'^(\s*)([-*+]|\d+\.)\s+(.+)$', stripped)
        
        if list_pattern:
            indent = list_pattern.group(1)
            marker = list_pattern.group(2)
            content = list_pattern.group(3)
            
            list_start = i
            list_items = [(indent, marker, content, line)]
            i += 1
            
            # Collect consecutive list items with same or deeper indentation
            while i < len(repaired_lines):
                next_line = repaired_lines[i]
                next_stripped = next_line.strip()
                
                # Check if continuation of previous item (indented, no marker)
                if next_stripped and not re.match(r'^(\s*)([-*+]|\d+\.)\s+', next_stripped):
                    # Might be continuation - check indentation
                    next_indent = len(next_line) - len(next_line.lstrip())
                    prev_indent = len(indent)
                    
                    if next_indent > prev_indent:
                        # Continuation line - append to previous item
                        list_items[-1] = (list_items[-1][0], list_items[-1][1], 
                                         list_items[-1][2] + " " + next_stripped, 
                                         list_items[-1][3] + "\n" + next_line)
                        i += 1
                        continue
                
                # Check if new list item
                next_match = re.match(r'^(\s*)([-*+]|\d+\.)\s+(.+)$', next_stripped)
                if next_match:
                    next_indent = next_match.group(1)
                    next_marker = next_match.group(2)
                    
                    # Same or deeper indentation - part of same list
                    if len(next_indent) >= len(indent):
                        list_items.append((next_indent, next_marker, next_match.group(3), next_line))
                        i += 1
                    else:
                        break
                else:
                    # Blank line or non-list content - end of list
                    if not next_stripped:
                        i += 1
                    break
            
            # Normalize list markers if needed
            if len(list_items) > 1:
                # Check if markers are inconsistent
                first_marker = list_items[0][1]
                markers_consistent = all(item[1] == first_marker for item in list_items)
                
                if not markers_consistent:
                    # Normalize to first marker type
                    for j, (ind, mark, cont, orig) in enumerate(list_items):
                        if mark != first_marker:
                            # Replace marker
                            new_line = ind + first_marker + " " + cont
                            repaired_lines[list_start + j] = new_line
                            
                            repair_records.append(RepairRecord(
                                repair_type="list_repair",
                                location=(list_start + j, list_start + j),
                                reason=f"Inconsistent list marker, normalized to '{first_marker}'",
                                original=orig,
                                repaired=new_line
                            ))
        else:
            i += 1
    
    return repaired_lines, repair_records


def repair_sections(lines: List[str]) -> Tuple[List[str], List[RepairRecord]]:
    """Repair missing or out-of-order headings by creating soft section boundaries.
    
    Uses blank lines, HRs (`---`), label-like lines, and bracket blocks.
    
    Args:
        lines: List of lines to repair
    
    Returns:
        Tuple[List[str], List[RepairRecord]]: Repaired lines and repair records
    """
    repaired_lines = lines.copy()
    repair_records = []
    
    i = 0
    while i < len(repaired_lines):
        line = repaired_lines[i].strip()
        
        # Detect potential section boundaries
        # 1. Horizontal rules
        if re.match(r'^---+$', line):
            # Already a section boundary
            i += 1
            continue
        
        # 2. Label-like lines (e.g., "Requirements:", "Summary:", etc.)
        label_pattern = re.match(r'^([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*):\s*$', line)
        if label_pattern and i > 0:
            # Check if previous line is blank (good section boundary)
            prev_line = repaired_lines[i - 1].strip() if i > 0 else ""
            
            if not prev_line:
                # Good section boundary already exists
                i += 1
                continue
            
            # Add blank line before label if missing
            if prev_line and not re.match(r'^#+\s+', prev_line):  # Not a heading
                repaired_lines.insert(i, "")
                repair_records.append(RepairRecord(
                    repair_type="section_repair",
                    location=(i, i),
                    reason=f"Added blank line before label '{line}' for section boundary",
                    original=prev_line + "\n" + line,
                    repaired=prev_line + "\n\n" + line
                ))
                i += 2  # Skip inserted line and current line
                continue
        
        # 3. Bracket blocks ([HEADER], [IMAGE]) - already section boundaries
        if line.startswith("[") and line.endswith("]"):
            i += 1
            continue
        
        i += 1
    
    return repaired_lines, repair_records


def apply_repair_mode(text: str) -> RepairResult:
    """Orchestrate all repair operations and compute structure confidence.
    
    Args:
        text: Input markdown text
    
    Returns:
        RepairResult: Repair result with repaired content, records, and confidence
    """
    # Normalize page markers first
    text = normalize_page_markers(text)
    
    # Split into lines for repair operations
    lines = text.split("\n")
    
    # Apply repairs in order
    repaired_lines, table_records = repair_tables(lines)
    repaired_lines, list_records = repair_lists(repaired_lines)
    repaired_lines, section_records = repair_sections(repaired_lines)
    
    # Combine all repair records
    repair_applied = {
        "table_repair": table_records,
        "list_repair": list_records,
        "section_repair": section_records
    }
    
    # Reconstruct text
    repaired_content = "\n".join(repaired_lines)
    
    # Compute structure confidence
    all_records = table_records + list_records + section_records
    from src.utils.tokens import compute_structure_confidence
    structure_confidence = compute_structure_confidence(
        [{"repair_type": r.repair_type} for r in all_records],
        repaired_content
    )
    
    return RepairResult(
        repaired_content=repaired_content,
        repair_applied=repair_applied,
        structure_confidence=structure_confidence
    )

