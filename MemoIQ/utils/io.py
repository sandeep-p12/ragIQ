"""I/O utilities for MemoIQ runs."""

import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional
from uuid import uuid4

from MemoIQ.config import MemoIQConfig


def create_run_directory(config: MemoIQConfig) -> Path:
    """
    Create a new run directory.
    
    Args:
        config: MemoIQConfig
        
    Returns:
        Path to run directory
    """
    run_id = str(uuid4())
    run_dir = config.runs_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    
    # Create subdirectories
    (run_dir / "template").mkdir(exist_ok=True)
    (run_dir / "reference_docs").mkdir(exist_ok=True)
    (run_dir / "outputs").mkdir(exist_ok=True)
    
    return run_dir


def save_template(run_dir: Path, template_path: str) -> Path:
    """Save template file to run directory."""
    template_dest = run_dir / "template" / Path(template_path).name
    shutil.copy2(template_path, template_dest)
    return template_dest


def save_reference_docs(run_dir: Path, doc_paths: list[str]) -> list[Path]:
    """Save reference documents to run directory."""
    saved_paths = []
    for doc_path in doc_paths:
        doc_dest = run_dir / "reference_docs" / Path(doc_path).name
        shutil.copy2(doc_path, doc_dest)
        saved_paths.append(doc_dest)
    return saved_paths


def save_json(run_dir: Path, filename: str, data: Any) -> Path:
    """Save JSON data to run directory."""
    file_path = run_dir / filename
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, default=str, ensure_ascii=False)
    return file_path


def load_json(run_dir: Path, filename: str) -> Any:
    """Load JSON data from run directory."""
    file_path = run_dir / filename
    if not file_path.exists():
        return None
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_draft(run_dir: Path, draft_path: str, version: int = 1) -> Path:
    """Save draft DOCX file to run directory."""
    draft_dest = run_dir / "outputs" / f"draft_v{version}.docx"
    shutil.copy2(draft_path, draft_dest)
    return draft_dest

