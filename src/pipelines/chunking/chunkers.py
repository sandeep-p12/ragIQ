"""Structure-first chunking + token-budget refinement with typed serialization."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

try:
    from langchain.text_splitter import RecursiveCharacterTextSplitter
except ImportError:
    # Fallback if langchain not available
    RecursiveCharacterTextSplitter = None

from src.pipelines.chunking import element_extractor
from src.pipelines.chunking.element_extractor import (
    Element,
    Heading,
    ImageBlockElement,
    ListElement,
    Paragraph,
    Table,
)
from src.schema.chunk import Chunk
from src.utils.ids import generate_chunk_id
from src.utils.tokens import count_tokens

# Chunk is now imported from schema


class StructureFirstChunker:
    """Create candidate chunks from elements (structure-preserving)."""
    
    def __init__(self, config):
        """Initialize chunker with config."""
        self.config = config
    
    def chunk(self, elements: List[Element], page_span: Tuple[int, int], 
              page_nos: List[int], doc_id: str, header_paths: Dict[int, str]) -> List[Chunk]:
        """Create candidate chunks from elements.
        
        Groups elements by section_label/header_path.
        Never splits inside list items, table rows, image blocks.
        
        Args:
            elements: List of elements
            page_span: Page span tuple
            page_nos: List of page numbers
            doc_id: Document ID
            header_paths: Mapping of element index to header path
        
        Returns:
            List[Chunk]: Candidate chunks
        """
        chunks = []
        current_group = []
        current_section = None
        group_has_heading = False  # Track if current group starts with a heading
        
        for i, element in enumerate(elements):
            header_path = header_paths.get(i)
            
            # Determine section label
            from src.pipelines.chunking.element_extractor import generate_section_label
            section_label = generate_section_label(element, header_path, page_span[0], i)
            
            # Check if current element is a heading
            is_heading = isinstance(element, Heading) or isinstance(element, element_extractor.CustomHeader)
            
            # If we encounter a new heading and current group has content, split
            # This ensures headings are always at the start of their content chunks
            if is_heading and current_group and group_has_heading:
                # Current group already has a heading, split it before adding new heading
                chunk = self._create_chunk_from_group(
                    current_group, current_section, page_span, page_nos, doc_id, header_paths
                )
                if chunk:
                    chunks.append(chunk)
                current_group = []
                group_has_heading = False
            
            # Add element to current group
            current_group.append((i, element))
            
            # Update section label and heading flag
            if is_heading:
                current_section = section_label
                group_has_heading = True
            elif current_section is None:
                # If no section set yet, use this element's section
                current_section = section_label
            # If group has a heading, keep grouping content with it regardless of section_label
            # This ensures headings are always with their following content
        
        # Create chunk from remaining group
        # CRITICAL: Always create chunk from remaining group to ensure no elements are lost
        # This is especially important for the last elements in a page block
        if current_group:
            chunk = self._create_chunk_from_group(
                current_group, current_section, page_span, page_nos, doc_id, header_paths
            )
            if chunk:
                chunks.append(chunk)
        # If current_group is empty but we're here, that's fine - all elements were chunked
        
        return chunks
    
    def _create_chunk_from_group(
        self, group: List[Tuple[int, Element]], section_label: str,
        page_span: Tuple[int, int], page_nos: List[int], doc_id: str,
        header_paths: Dict[int, str]
    ) -> Optional[Chunk]:
        """Create chunk from element group.
        
        Always creates a chunk, even for standalone headings, to ensure no content is lost.
        """
        if not group:
            return None
        
        # Ensure we always create chunks, even for single headings
        # This prevents headings from being lost
        
        # Get header path from first element
        first_idx, first_element = group[0]
        header_path = header_paths.get(first_idx)
        
        # Determine element type (use first element's type, or "mixed")
        element_types = [type(e).__name__.lower() for _, e in group]
        if len(set(element_types)) == 1:
            element_type = element_types[0]
        else:
            element_type = "mixed"
        
        # Build raw markdown fragment
        raw_parts = []
        for _, element in group:
            if isinstance(element, Paragraph):
                raw_parts.append(element.text)
            elif isinstance(element, List):
                # Reconstruct list with markers
                # For ordered lists, use numbered markers (will be 1., 2., etc.)
                # Note: Original lettered markers (A., B., etc.) are preserved in element.items content
                if element.ordered:
                    marker = "1. "  # Default to numbered
                else:
                    marker = "- "
                for idx, item in enumerate(element.items):
                    if element.ordered:
                        marker = f"{idx + 1}. "
                    raw_parts.append(element.nesting + marker + item)
            elif isinstance(element, Table):
                raw_parts.append(element.raw_md)
            elif isinstance(element, ImageBlockElement):
                raw_parts.append(element.raw_text)
            elif isinstance(element, Heading):
                level = "#" * element.level
                raw_parts.append(f"{level} {element.text}")
            elif isinstance(element, element_extractor.CustomHeader):
                raw_parts.append(f"[HEADER]{element.text}[/HEADER]")
        
        raw_md_fragment = "\n\n".join(raw_parts)
        
        # Create metadata
        metadata = {
            "element_count": len(group),
            "element_types": element_types,
        }
        
        # Add element-specific metadata
        for idx, element in group:
            if isinstance(element, Table):
                metadata["table_signature"] = element.signature
                metadata["table_rows"] = len(element.rows)
                if element.header_row:
                    metadata["table_header"] = element.header_row
            elif isinstance(element, List):
                metadata["list_items"] = len(element.items)
                metadata["list_ordered"] = element.ordered
        
        # Generate text_for_embedding (will be refined by serialization)
        text_for_embedding = serialize_for_embedding(group, section_label, header_path, doc_id, page_span)
        
        # Generate chunk ID
        chunk_id = generate_chunk_id(raw_md_fragment, {
            "doc_id": doc_id,
            "page_span": page_span,
            "section_label": section_label,
            "element_type": element_type
        })
        
        # Count tokens
        token_count = count_tokens(text_for_embedding)
        
        # Get node_id from first element
        node_id = group[0][1].node_id if hasattr(group[0][1], 'node_id') else ""
        
        # Track line positions from elements (min start, max end)
        line_starts = []
        line_ends = []
        for _, element in group:
            if hasattr(element, 'line_start') and element.line_start is not None:
                line_starts.append(element.line_start)
            if hasattr(element, 'line_end') and element.line_end is not None:
                line_ends.append(element.line_end)
        
        line_start = min(line_starts) if line_starts else None
        line_end = max(line_ends) if line_ends else None
        
        return Chunk(
            chunk_id=chunk_id,
            doc_id=doc_id,
            page_span=page_span,
            page_nos=page_nos,
            header_path=header_path,
            section_label=section_label,
            element_type=element_type,
            raw_md_fragment=raw_md_fragment,
            text_for_embedding=text_for_embedding,
            metadata=metadata,
            token_count=token_count,
            node_id=node_id,
            line_start=line_start,
            line_end=line_end
        )


class TokenBudgetRefiner:
    """Refine chunks by token budget: split oversized, merge tiny."""
    
    def __init__(self, config):
        """Initialize refiner with config."""
        self.config = config
        self.prose_refiner = ProseRefiner(config)
        self.list_refiner = ListRefiner(config)
        self.table_refiner = TableRefiner(config)
        self.image_refiner = ImageRefiner(config)
    
    def refine(self, chunks: List[Chunk]) -> List[Chunk]:
        """Refine chunks by token budget.
        
        Splits oversized chunks safely by type.
        Merges tiny adjacent chunks when safe.
        
        Args:
            chunks: List of candidate chunks
        
        Returns:
            List[Chunk]: Refined chunks
        """
        refined = []
        
        for chunk in chunks:
            # Check if chunk is oversized
            if chunk.token_count > self.config.max_chunk_tokens_hard:
                # Split based on element type
                split_chunks = self._split_chunk(chunk)
                refined.extend(split_chunks)
            else:
                refined.append(chunk)
        
        # Merge tiny chunks
        refined = self._merge_tiny_chunks(refined)
        
        return refined
    
    def _split_chunk(self, chunk: Chunk) -> List[Chunk]:
        """Split oversized chunk based on element type."""
        element_type = chunk.element_type.lower()
        
        if element_type == "paragraph" or element_type == "mixed":
            return self.prose_refiner.split(chunk)
        elif element_type == "list":
            return self.list_refiner.split(chunk)
        elif element_type == "table":
            return self.table_refiner.split(chunk)
        elif element_type == "imageblock":
            return self.image_refiner.split(chunk)
        else:
            # Default: use prose refiner
            return self.prose_refiner.split(chunk)
    
    def _merge_tiny_chunks(self, chunks: List[Chunk]) -> List[Chunk]:
        """Merge tiny adjacent chunks when safe.
        
        Only merges when:
        - Same page_span (or adjacent pages)
        - Compatible element types
        - Combined size within limits
        """
        if not chunks:
            return chunks
        
        merged = []
        i = 0
        # Use min_chunk_tokens if available, otherwise fall back to prose_target_tokens // 2
        tiny_threshold = getattr(self.config, 'min_chunk_tokens', None) or (self.config.prose_target_tokens // 2)
        
        while i < len(chunks):
            current = chunks[i]
            
            # Check if current chunk is tiny
            is_tiny = current.token_count < tiny_threshold
            
            if is_tiny and i < len(chunks) - 1:
                next_chunk = chunks[i + 1]
                
                # Check if merge is safe
                if self._can_merge(current, next_chunk):
                    # Merge chunks
                    merged_chunk = self._merge_two_chunks(current, next_chunk)
                    merged.append(merged_chunk)
                    i += 2
                    continue
            
            merged.append(current)
            i += 1
        
        return merged
    
    def _can_merge(self, chunk1: Chunk, chunk2: Chunk) -> bool:
        """Check if two chunks can be safely merged.
        
        Allows merging chunks from adjacent pages if they're tiny/small,
        to avoid having many very small chunks. More lenient for small chunks.
        """
        # Same page span OR adjacent pages (for tiny/small chunks)
        page_diff = abs(chunk1.page_span[0] - chunk2.page_span[0])
        # Allow merging across up to 2 pages for very small chunks
        max_page_diff = 2 if (chunk1.token_count < 100 or chunk2.token_count < 100) else 1
        if page_diff > max_page_diff:
            return False
        
        # Combined size within limits
        combined_tokens = chunk1.token_count + chunk2.token_count
        if combined_tokens > self.config.max_chunk_tokens_hard:
            return False
        
        # Compatible element types (both prose, or both same type, or one is mixed)
        type1 = chunk1.element_type.lower()
        type2 = chunk2.element_type.lower()
        
        # If either is mixed, allow merge (mixed can contain anything)
        if "mixed" in [type1, type2]:
            return True
        
        # If both are the same type, allow merge
        if type1 == type2:
            return True
        
        # If both are prose-like, allow merge
        prose_types = ["paragraph", "heading", "customheader"]
        if type1 in prose_types and type2 in prose_types:
            return True
        
        # For very small chunks (< 100 tokens), be more lenient
        # Allow merging table with prose, list with prose, etc.
        if chunk1.token_count < 100 or chunk2.token_count < 100:
            # Allow merging any small chunk with prose
            if type1 in prose_types or type2 in prose_types:
                return True
        
        # Don't merge incompatible types (e.g., table with list) for larger chunks
        return False
    
    def _merge_two_chunks(self, chunk1: Chunk, chunk2: Chunk) -> Chunk:
        """Merge two chunks into one."""
        # Combine raw markdown
        raw_md = chunk1.raw_md_fragment + "\n\n" + chunk2.raw_md_fragment
        
        # Combine text for embedding
        text_embed = chunk1.text_for_embedding + "\n\n" + chunk2.text_for_embedding
        
        # Combine metadata
        metadata = {**chunk1.metadata, **chunk2.metadata}
        metadata["merged_from"] = [chunk1.chunk_id, chunk2.chunk_id]
        
        # Choose the best section label (prefer meaningful labels over "Untitled")
        # Priority: 1) Non-Untitled labels, 2) Labels with header_path, 3) First chunk's label
        section_label = chunk1.section_label
        
        # Prefer chunk2's label if it's meaningful and chunk1's is "Untitled"
        if chunk1.section_label.startswith("Untitled") and not chunk2.section_label.startswith("Untitled"):
            section_label = chunk2.section_label
        # If both are meaningful, prefer the one with header_path
        elif not chunk1.section_label.startswith("Untitled") and not chunk2.section_label.startswith("Untitled"):
            if chunk2.header_path and not chunk1.header_path:
                section_label = chunk2.section_label
        # If both are Untitled, use the first one (to avoid long concatenated names)
        # Keep chunk1's label as default
        
        # Generate new chunk ID
        chunk_id = generate_chunk_id(raw_md, {
            "doc_id": chunk1.doc_id,
            "page_span": chunk1.page_span,
            "section_label": section_label
        })
        
        # Merge line positions (min start, max end)
        line_start = None
        line_end = None
        if chunk1.line_start is not None or chunk2.line_start is not None:
            starts = [s for s in [chunk1.line_start, chunk2.line_start] if s is not None]
            line_start = min(starts) if starts else None
        if chunk1.line_end is not None or chunk2.line_end is not None:
            ends = [e for e in [chunk1.line_end, chunk2.line_end] if e is not None]
            line_end = max(ends) if ends else None
        
        return Chunk(
            chunk_id=chunk_id,
            doc_id=chunk1.doc_id,
            page_span=chunk1.page_span,
            page_nos=chunk1.page_nos,
            header_path=chunk1.header_path or chunk2.header_path,
            section_label=section_label,
            element_type="mixed",
            raw_md_fragment=raw_md,
            text_for_embedding=text_embed,
            metadata=metadata,
            parent_id=chunk1.parent_id or chunk2.parent_id,
            token_count=count_tokens(text_embed),
            node_id=chunk1.node_id,
            line_start=line_start,
            line_end=line_end
        )


class ProseRefiner:
    """Refine prose chunks: sentence + windowed sentences, LangChain fallback."""
    
    def __init__(self, config):
        """Initialize prose refiner."""
        self.config = config
        if RecursiveCharacterTextSplitter:
            self.langchain_splitter = RecursiveCharacterTextSplitter(
                chunk_size=config.prose_target_tokens * 4,  # Approximate chars
                chunk_overlap=config.prose_overlap_tokens * 4
            )
        else:
            self.langchain_splitter = None
    
    def split(self, chunk: Chunk) -> List[Chunk]:
        """Split prose chunk using sentence chunking + window."""
        text = chunk.raw_md_fragment
        
        # Try sentence splitting first
        sentences = re.split(r'([.!?]+\s+)', text)
        sentences = [s1 + s2 for s1, s2 in zip(sentences[::2], sentences[1::2])] if len(sentences) > 1 else [text]
        
        # If sentences work, use windowed approach
        if len(sentences) > 1:
            return self._split_by_sentences(chunk, sentences)
        else:
            # Fallback to LangChain splitter
            return self._split_with_langchain(chunk)
    
    def _split_by_sentences(self, chunk: Chunk, sentences: List[str]) -> List[Chunk]:
        """Split chunk by sentences with windowing."""
        chunks = []
        window_size = self.config.sentence_window_size
        
        i = 0
        while i < len(sentences):
            # Collect sentences up to target size
            current_text = []
            current_tokens = 0
            
            # Add sentences with window overlap
            start_idx = max(0, i - window_size)
            for j in range(start_idx, min(i + self.config.prose_target_tokens // 50, len(sentences))):
                sent = sentences[j]
                sent_tokens = count_tokens(sent)
                
                if current_tokens + sent_tokens > self.config.prose_target_tokens and current_text:
                    break
                
                current_text.append(sent)
                current_tokens += sent_tokens
            
            if current_text:
                raw_md = "".join(current_text)
                text_embed = serialize_for_embedding_prose(raw_md, chunk.section_label, 
                                                           chunk.header_path, chunk.doc_id, chunk.page_span)
                
                chunk_id = generate_chunk_id(raw_md, {
                    "doc_id": chunk.doc_id,
                    "page_span": chunk.page_span,
                    "section_label": chunk.section_label,
                    "split_index": i
                })
                
                chunks.append(Chunk(
                    chunk_id=chunk_id,
                    doc_id=chunk.doc_id,
                    page_span=chunk.page_span,
                    page_nos=chunk.page_nos,
                    header_path=chunk.header_path,
                    section_label=chunk.section_label,
                    element_type=chunk.element_type,
                    raw_md_fragment=raw_md,
                    text_for_embedding=text_embed,
                    metadata={**chunk.metadata, "split_from": chunk.chunk_id, "sentence_range": (start_idx, i)},
                    parent_id=chunk.parent_id,
                    token_count=count_tokens(text_embed),
                    node_id=chunk.node_id,
                    line_start=chunk.line_start if i == 0 else None,  # Preserve start for first split
                    line_end=chunk.line_end if i + len(current_text) >= len(sentences) else None  # Preserve end for last split
                ))
            
            i += len(current_text) if current_text else 1
        
        return chunks
    
    def _split_with_langchain(self, chunk: Chunk) -> List[Chunk]:
        """Split chunk using LangChain RecursiveCharacterTextSplitter."""
        if not self.langchain_splitter:
            # Fallback to simple splitting if langchain not available
            return self._split_simple(chunk)
        splits = self.langchain_splitter.split_text(chunk.raw_md_fragment)
        
        chunks = []
        for idx, split_text in enumerate(splits):
            text_embed = serialize_for_embedding_prose(split_text, chunk.section_label,
                                                      chunk.header_path, chunk.doc_id, chunk.page_span)
            
            chunk_id = generate_chunk_id(split_text, {
                "doc_id": chunk.doc_id,
                "page_span": chunk.page_span,
                "section_label": chunk.section_label,
                "split_index": idx
            })
            
            chunks.append(Chunk(
                chunk_id=chunk_id,
                doc_id=chunk.doc_id,
                page_span=chunk.page_span,
                page_nos=chunk.page_nos,
                header_path=chunk.header_path,
                section_label=chunk.section_label,
                element_type=chunk.element_type,
                raw_md_fragment=split_text,
                text_for_embedding=text_embed,
                metadata={**chunk.metadata, "split_from": chunk.chunk_id, "langchain_split": True},
                parent_id=chunk.parent_id,
                token_count=count_tokens(text_embed),
                node_id=chunk.node_id,
                line_start=chunk.line_start if idx == 0 else None,  # Preserve start for first split
                line_end=chunk.line_end if idx == len(splits) - 1 else None  # Preserve end for last split
            ))
        
        return chunks
    
    def _split_simple(self, chunk: Chunk) -> List[Chunk]:
        """Simple fallback splitting when langchain not available."""
        # Split by paragraphs
        paragraphs = chunk.raw_md_fragment.split("\n\n")
        chunks = []
        
        current_text = []
        current_tokens = 0
        chunk_count = 0
        
        for para in paragraphs:
            para_tokens = count_tokens(para)
            if current_tokens + para_tokens > self.config.prose_target_tokens and current_text:
                # Create chunk
                raw_md = "\n\n".join(current_text)
                text_embed = serialize_for_embedding_prose(raw_md, chunk.section_label,
                                                          chunk.header_path, chunk.doc_id, chunk.page_span)
                chunk_id = generate_chunk_id(raw_md, {
                    "doc_id": chunk.doc_id,
                    "page_span": chunk.page_span,
                    "section_label": chunk.section_label
                })
                chunks.append(Chunk(
                    chunk_id=chunk_id,
                    doc_id=chunk.doc_id,
                    page_span=chunk.page_span,
                    page_nos=chunk.page_nos,
                    header_path=chunk.header_path,
                    section_label=chunk.section_label,
                    element_type=chunk.element_type,
                    raw_md_fragment=raw_md,
                    text_for_embedding=text_embed,
                    metadata={**chunk.metadata, "split_from": chunk.chunk_id, "simple_split": True},
                    parent_id=chunk.parent_id,
                    token_count=count_tokens(text_embed),
                    node_id=chunk.node_id,
                    line_start=chunk.line_start if chunk_count == 0 else None,  # Preserve start for first split
                    line_end=None  # Can't determine end until we know if this is the last
                ))
                chunk_count += 1
                current_text = [para]
                current_tokens = para_tokens
            else:
                current_text.append(para)
                current_tokens += para_tokens
        
        # Add remaining
        if current_text:
            raw_md = "\n\n".join(current_text)
            text_embed = serialize_for_embedding_prose(raw_md, chunk.section_label,
                                                      chunk.header_path, chunk.doc_id, chunk.page_span)
            chunk_id = generate_chunk_id(raw_md, {
                "doc_id": chunk.doc_id,
                "page_span": chunk.page_span,
                "section_label": chunk.section_label
            })
            chunks.append(Chunk(
                chunk_id=chunk_id,
                doc_id=chunk.doc_id,
                page_span=chunk.page_span,
                page_nos=chunk.page_nos,
                header_path=chunk.header_path,
                section_label=chunk.section_label,
                element_type=chunk.element_type,
                raw_md_fragment=raw_md,
                text_for_embedding=text_embed,
                metadata={**chunk.metadata, "split_from": chunk.chunk_id, "simple_split": True},
                parent_id=chunk.parent_id,
                token_count=count_tokens(text_embed),
                node_id=chunk.node_id,
                line_start=chunk.line_start if chunk_count == 0 else None,  # Preserve start if this is the only chunk
                line_end=chunk.line_end  # Preserve end for last split
            ))
        
        return chunks


class ListRefiner:
    """Refine list chunks: by item groups with overlap."""
    
    def __init__(self, config):
        """Initialize list refiner."""
        self.config = config
    
    def split(self, chunk: Chunk) -> List[Chunk]:
        """Split list chunk by item groups."""
        # Extract items from metadata or parse from raw_md
        items = chunk.metadata.get("list_items", [])
        if not items:
            # Parse from raw_md
            lines = chunk.raw_md_fragment.split("\n")
            items = [line.strip() for line in lines if line.strip() and re.match(r'^\s*[-*+]|\d+\.', line.strip())]
        
        if not items:
            return [chunk]
        
        chunks = []
        items_per_chunk = self.config.list_items_per_chunk
        overlap = self.config.list_item_overlap
        
        i = 0
        while i < len(items):
            # Get items for this chunk
            end_idx = min(i + items_per_chunk, len(items))
            chunk_items = items[i:end_idx]
            
            # Reconstruct list markdown
            raw_md = "\n".join(chunk_items)
            text_embed = serialize_for_embedding_list(chunk_items, chunk.section_label,
                                                     chunk.header_path, chunk.doc_id, chunk.page_span)
            
            chunk_id = generate_chunk_id(raw_md, {
                "doc_id": chunk.doc_id,
                "page_span": chunk.page_span,
                "section_label": chunk.section_label,
                "item_range": (i, end_idx)
            })
            
            chunks.append(Chunk(
                chunk_id=chunk_id,
                doc_id=chunk.doc_id,
                page_span=chunk.page_span,
                page_nos=chunk.page_nos,
                header_path=chunk.header_path,
                section_label=chunk.section_label,
                element_type="list",
                raw_md_fragment=raw_md,
                text_for_embedding=text_embed,
                metadata={**chunk.metadata, "split_from": chunk.chunk_id, "item_range": (i, end_idx)},
                parent_id=chunk.parent_id,
                token_count=count_tokens(text_embed),
                node_id=chunk.node_id,
                line_start=chunk.line_start if i == 0 else None,  # Preserve start for first split
                line_end=chunk.line_end if end_idx >= len(items) else None  # Preserve end for last split
            ))
            
            # Move forward with overlap
            i += items_per_chunk - overlap
        
        return chunks


class TableRefiner:
    """Refine table chunks: by row groups with header repetition."""
    
    def __init__(self, config):
        """Initialize table refiner."""
        self.config = config
    
    def split(self, chunk: Chunk) -> List[Chunk]:
        """Split table chunk by row groups."""
        # Extract table data from metadata
        header_row = chunk.metadata.get("table_header", "")
        rows = chunk.metadata.get("table_rows", [])
        
        if not rows:
            # Parse from raw_md
            lines = chunk.raw_md_fragment.split("\n")
            header_row = None
            rows = []
            for line in lines:
                if '|' in line:
                    if not header_row and not re.match(r'^[\s\-\|:]+$', line.strip()):
                        header_row = line.strip()
                    elif header_row and not re.match(r'^[\s\-\|:]+$', line.strip()):
                        rows.append(line.strip())
        
        if not rows:
            return [chunk]
        
        chunks = []
        rows_per_chunk = self.config.table_rows_per_chunk
        overlap = self.config.table_row_overlap
        
        i = 0
        while i < len(rows):
            end_idx = min(i + rows_per_chunk, len(rows))
            chunk_rows = rows[i:end_idx]
            
            # Reconstruct table with header
            table_lines = []
            if header_row:
                table_lines.append(header_row)
                # Add separator
                num_cols = len(header_row.split('|')) - 1
                separator = '|' + '|'.join([' --- '] * num_cols) + '|'
                table_lines.append(separator)
            table_lines.extend(chunk_rows)
            
            raw_md = "\n".join(table_lines)
            text_embed = serialize_for_embedding_table(chunk_rows, header_row, chunk.metadata.get("table_signature", ""),
                                                       chunk.section_label, chunk.header_path, chunk.doc_id, chunk.page_span,
                                                       (i, end_idx))
            
            chunk_id = generate_chunk_id(raw_md, {
                "doc_id": chunk.doc_id,
                "page_span": chunk.page_span,
                "section_label": chunk.section_label,
                "row_range": (i, end_idx)
            })
            
            chunks.append(Chunk(
                chunk_id=chunk_id,
                doc_id=chunk.doc_id,
                page_span=chunk.page_span,
                page_nos=chunk.page_nos,
                header_path=chunk.header_path,
                section_label=chunk.section_label,
                element_type="table",
                raw_md_fragment=raw_md,
                text_for_embedding=text_embed,
                metadata={**chunk.metadata, "split_from": chunk.chunk_id, "row_range": (i, end_idx)},
                parent_id=chunk.parent_id,
                token_count=count_tokens(text_embed),
                node_id=chunk.node_id,
                line_start=chunk.line_start if i == 0 else None,  # Preserve start for first split
                line_end=chunk.line_end if end_idx >= len(rows) else None  # Preserve end for last split
            ))
            
            i += rows_per_chunk - overlap
        
        return chunks


class ImageRefiner:
    """Refine image chunks: atomic or paragraph-boundary split."""
    
    def __init__(self, config):
        """Initialize image refiner."""
        self.config = config
    
    def split(self, chunk: Chunk) -> List[Chunk]:
        """Split image chunk (usually atomic, but can split if extremely long)."""
        # Image blocks are usually atomic
        # Only split if extremely long (by paragraph boundaries inside [IMAGE])
        if chunk.token_count < self.config.max_chunk_tokens_hard * 2:
            return [chunk]
        
        # Split by paragraphs inside [IMAGE] block
        extracted_text = chunk.metadata.get("extracted_text", "")
        if not extracted_text:
            return [chunk]
        
        paragraphs = extracted_text.split("\n\n")
        if len(paragraphs) <= 1:
            return [chunk]
        
        # Create chunks for each paragraph (preserving [IMAGE] markers)
        chunks = []
        for idx, para in enumerate(paragraphs):
            raw_md = f"[IMAGE]\n{para}\n[/IMAGE]"
            text_embed = serialize_for_embedding_image(para, chunk.section_label,
                                                      chunk.header_path, chunk.doc_id, chunk.page_span)
            
            chunk_id = generate_chunk_id(raw_md, {
                "doc_id": chunk.doc_id,
                "page_span": chunk.page_span,
                "section_label": chunk.section_label,
                "paragraph_index": idx
            })
            
            chunks.append(Chunk(
                chunk_id=chunk_id,
                doc_id=chunk.doc_id,
                page_span=chunk.page_span,
                page_nos=chunk.page_nos,
                header_path=chunk.header_path,
                section_label=chunk.section_label,
                element_type="imageblock",
                raw_md_fragment=raw_md,
                text_for_embedding=text_embed,
                metadata={**chunk.metadata, "split_from": chunk.chunk_id, "paragraph_index": idx},
                parent_id=chunk.parent_id,
                token_count=count_tokens(text_embed),
                node_id=chunk.node_id,
                line_start=chunk.line_start if idx == 0 else None,  # Preserve start for first split
                line_end=chunk.line_end if idx == len(paragraphs) - 1 else None  # Preserve end for last split
            ))
        
        return chunks


def serialize_for_embedding(
    group: List[Tuple[int, Element]],
    section_label: str,
    header_path: Optional[str],
    doc_id: str,
    page_span: Tuple[int, int]
) -> str:
    """Serialize element group for embedding.
    
    Type-aware serialization with minimal context.
    """
    if not group:
        return ""
    
    # Build context header
    context_parts = [f"Document: {doc_id}"]
    if page_span[0] == page_span[1]:
        context_parts.append(f"Page: {page_span[0]}")
    else:
        context_parts.append(f"Pages: {page_span[0]}-{page_span[1]}")
    
    if header_path:
        context_parts.append(f"Section: {header_path}")
    elif section_label:
        context_parts.append(f"Section: {section_label}")
    
    context_header = " | ".join(context_parts)
    
    # Serialize elements based on type
    content_parts = []
    for _, element in group:
        if isinstance(element, Paragraph):
            content_parts.append(element.text)
        elif isinstance(element, List):
            marker = "1. " if element.ordered else "- "
            for item in element.items:
                content_parts.append(marker + item)
        elif isinstance(element, Table):
            if element.header_row:
                content_parts.append(f"Table Header: {element.header_row}")
            for row in element.rows:
                content_parts.append(f"Row: {row}")
        elif isinstance(element, ImageBlockElement):
            content_parts.append(f"[IMAGE]\n{element.extracted_text}\n[/IMAGE]")
        elif isinstance(element, Heading):
            content_parts.append(f"{'#' * element.level} {element.text}")
        elif isinstance(element, element_extractor.CustomHeader):
            content_parts.append(f"[HEADER]{element.text}[/HEADER]")
    
    content = "\n".join(content_parts)
    return f"{context_header}\n\n{content}"


def serialize_for_embedding_prose(
    text: str,
    section_label: str,
    header_path: Optional[str],
    doc_id: str,
    page_span: Tuple[int, int]
) -> str:
    """Serialize prose for embedding."""
    context_parts = [f"Document: {doc_id}"]
    if page_span[0] == page_span[1]:
        context_parts.append(f"Page: {page_span[0]}")
    else:
        context_parts.append(f"Pages: {page_span[0]}-{page_span[1]}")
    
    if header_path:
        context_parts.append(f"Section: {header_path}")
    elif section_label:
        context_parts.append(f"Section: {section_label}")
    
    context_header = " | ".join(context_parts)
    return f"{context_header}\n\n{text}"


def serialize_for_embedding_list(
    items: List[str],
    section_label: str,
    header_path: Optional[str],
    doc_id: str,
    page_span: Tuple[int, int]
) -> str:
    """Serialize list for embedding."""
    context_parts = [f"Document: {doc_id}"]
    if page_span[0] == page_span[1]:
        context_parts.append(f"Page: {page_span[0]}")
    else:
        context_parts.append(f"Pages: {page_span[0]}-{page_span[1]}")
    
    if header_path:
        context_parts.append(f"Section: {header_path}")
    elif section_label:
        context_parts.append(f"Section: {section_label}")
    
    context_header = " | ".join(context_parts)
    list_text = "\n".join(items)
    return f"{context_header}\n\nList Items:\n{list_text}"


def serialize_for_embedding_table(
    rows: List[str],
    header_row: Optional[str],
    table_signature: str,
    section_label: str,
    header_path: Optional[str],
    doc_id: str,
    page_span: Tuple[int, int],
    row_range: Tuple[int, int]
) -> str:
    """Serialize table for embedding."""
    context_parts = [f"Document: {doc_id}"]
    if page_span[0] == page_span[1]:
        context_parts.append(f"Page: {page_span[0]}")
    else:
        context_parts.append(f"Pages: {page_span[0]}-{page_span[1]}")
    
    if header_path:
        context_parts.append(f"Section: {header_path}")
    elif section_label:
        context_parts.append(f"Section: {section_label}")
    
    context_parts.append(f"Table: {table_signature}")
    context_parts.append(f"Rows: {row_range[0]}-{row_range[1]}")
    
    context_header = " | ".join(context_parts)
    
    table_parts = []
    if header_row:
        table_parts.append(f"Header: {header_row}")
    for row in rows:
        table_parts.append(f"Row: {row}")
    
    table_text = "\n".join(table_parts)
    return f"{context_header}\n\n{table_text}"


def serialize_for_embedding_image(
    extracted_text: str,
    section_label: str,
    header_path: Optional[str],
    doc_id: str,
    page_span: Tuple[int, int]
) -> str:
    """Serialize image block for embedding."""
    context_parts = [f"Document: {doc_id}"]
    if page_span[0] == page_span[1]:
        context_parts.append(f"Page: {page_span[0]}")
    else:
        context_parts.append(f"Pages: {page_span[0]}-{page_span[1]}")
    
    if header_path:
        context_parts.append(f"Section: {header_path}")
    elif section_label:
        context_parts.append(f"Section: {section_label}")
    
    context_header = " | ".join(context_parts)
    return f"{context_header}\n\n[IMAGE]\n{extracted_text}\n[/IMAGE]"



