"""Local file-based chunk store implementation."""

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.core.interfaces import ChunkStore


class LocalChunkStore(ChunkStore):
    """Local file-based chunk store for development."""
    
    def __init__(self, base_dir: str = "src/data/retrieval"):
        """Initialize local chunk store.
        
        Args:
            base_dir: Base directory for chunk storage (default: "src/data/retrieval")
        """
        self.base_dir = Path(base_dir)
        self.chunks_dir = self.base_dir / "chunks"
        self.parents_dir = self.base_dir / "parents"
        
        # Ensure directories exist
        self.chunks_dir.mkdir(parents=True, exist_ok=True)
        self.parents_dir.mkdir(parents=True, exist_ok=True)
    
    def put_chunks(
        self,
        children: List[Dict[str, Any]],
        parents: List[Dict[str, Any]]
    ) -> None:
        """Store chunks and parents to disk.
        
        Args:
            children: List of child chunk dictionaries
            parents: List of parent chunk dictionaries
        """
        # Store children
        for chunk in children:
            doc_id = chunk.get("doc_id", "unknown")
            chunk_id = chunk.get("chunk_id")
            if not chunk_id:
                continue
            
            doc_dir = self.chunks_dir / doc_id
            doc_dir.mkdir(parents=True, exist_ok=True)
            
            chunk_file = doc_dir / f"{chunk_id}.json"
            self._write_chunk_atomic(chunk_file, chunk)
        
        # Store parents
        for parent in parents:
            doc_id = parent.get("doc_id", "unknown")
            chunk_id = parent.get("chunk_id")
            if not chunk_id:
                continue
            
            doc_dir = self.parents_dir / doc_id
            doc_dir.mkdir(parents=True, exist_ok=True)
            
            parent_file = doc_dir / f"{chunk_id}.json"
            self._write_chunk_atomic(parent_file, parent)
    
    def get_chunk(
        self,
        doc_id: str,
        chunk_id: str,
        is_parent: bool = False
    ) -> Optional[Dict[str, Any]]:
        """Get a single chunk from disk.
        
        Args:
            doc_id: Document ID
            chunk_id: Chunk ID
            is_parent: Whether this is a parent chunk
            
        Returns:
            Chunk dictionary or None if not found
        """
        base_dir = self.parents_dir if is_parent else self.chunks_dir
        chunk_file = base_dir / doc_id / f"{chunk_id}.json"
        
        if not chunk_file.exists():
            return None
        
        try:
            with open(chunk_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            return None
    
    def get_chunks_bulk(
        self,
        keys: List[tuple[str, str, bool]]  # (doc_id, chunk_id, is_parent)
    ) -> List[Optional[Dict[str, Any]]]:
        """Get multiple chunks in bulk.
        
        Args:
            keys: List of (doc_id, chunk_id, is_parent) tuples
            
        Returns:
            List of chunk dictionaries (None for missing chunks)
        """
        results = []
        for doc_id, chunk_id, is_parent in keys:
            chunk = self.get_chunk(doc_id, chunk_id, is_parent)
            results.append(chunk)
        return results
    
    def _write_chunk_atomic(self, file_path: Path, chunk: Dict[str, Any]) -> None:
        """Write chunk to file atomically (write to temp, then rename).
        
        Args:
            file_path: Target file path
            chunk: Chunk dictionary to write
        """
        # Write to temporary file first
        temp_file = file_path.with_suffix(".tmp")
        
        try:
            with open(temp_file, "w", encoding="utf-8") as f:
                json.dump(chunk, f, ensure_ascii=False, indent=2)
            
            # Atomic rename
            temp_file.replace(file_path)
        except Exception:
            # Clean up temp file on error
            if temp_file.exists():
                temp_file.unlink()
            raise

