"""Utility functions for the RAG system."""

from src.utils.bbox import calculate_iou, bbox_distance
from src.utils.checkpoint import Checkpoint
from src.utils.env import get_openai_api_key, load_env
from src.utils.exceptions import (
    CheckpointError,
    ConfigurationError,
    LayoutError,
    OCRError,
    ParseForgeException,
    ParserError,
    StrategyException,
    TableError,
)
from src.utils.ids import generate_chunk_id
from src.utils.io import format_citation, load_jsonl, save_jsonl
from src.utils.memory import calculate_batch_size, get_available_memory, get_vram_usage
from src.utils.tokens import compute_structure_confidence, count_tokens
from src.utils.ui import display_error, format_progress_message, format_stage_output

__all__ = [
    # Environment
    "load_env",
    "get_openai_api_key",
    # Tokens
    "count_tokens",
    "compute_structure_confidence",
    # IDs
    "generate_chunk_id",
    # I/O
    "load_jsonl",
    "save_jsonl",
    "format_citation",
    # BBox
    "calculate_iou",
    "bbox_distance",
    # Checkpoint
    "Checkpoint",
    # Exceptions
    "ParseForgeException",
    "StrategyException",
    "OCRError",
    "LayoutError",
    "TableError",
    "ParserError",
    "CheckpointError",
    "ConfigurationError",
    # Memory
    "get_available_memory",
    "get_vram_usage",
    "calculate_batch_size",
    # UI
    "display_error",
    "format_progress_message",
    "format_stage_output",
]

