"""Checkpoint/resume mechanism for ParseForge."""

import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional

from src.utils.exceptions import CheckpointError

logger = logging.getLogger(__name__)


class Checkpoint:
    """Checkpoint manager for parsing state."""

    def __init__(self, checkpoint_dir: Path):
        """
        Initialize checkpoint manager.

        Args:
            checkpoint_dir: Directory for checkpoint files
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def save(
        self,
        document_path: str,
        last_batch: int,
        last_page: int,
        strategy: str,
        batches: list[Dict[str, Any]],
        checkpoint_id: Optional[str] = None,
    ) -> str:
        """
        Save checkpoint.

        Args:
            document_path: Path to document being parsed
            last_batch: Last completed batch index
            last_page: Last completed page index
            strategy: Current strategy
            batches: List of batch statuses
            checkpoint_id: Optional checkpoint ID

        Returns:
            Checkpoint file path
        """
        if checkpoint_id is None:
            checkpoint_id = Path(document_path).stem

        checkpoint_data = {
            "version": "1.0",
            "document_path": document_path,
            "last_batch": last_batch,
            "last_page": last_page,
            "strategy": strategy,
            "batches": batches,
        }

        checkpoint_file = self.checkpoint_dir / f"{checkpoint_id}.json"

        try:
            with open(checkpoint_file, "w") as f:
                json.dump(checkpoint_data, f, indent=2)
            logger.info(f"Saved checkpoint to {checkpoint_file}")
            return str(checkpoint_file)
        except Exception as e:
            raise CheckpointError(f"Failed to save checkpoint: {e}") from e

    def load(self, checkpoint_id: str) -> Dict[str, Any]:
        """
        Load checkpoint.

        Args:
            checkpoint_id: Checkpoint ID

        Returns:
            Checkpoint data
        """
        checkpoint_file = self.checkpoint_dir / f"{checkpoint_id}.json"

        if not checkpoint_file.exists():
            raise CheckpointError(f"Checkpoint not found: {checkpoint_file}")

        try:
            with open(checkpoint_file, "r") as f:
                data = json.load(f)
            logger.info(f"Loaded checkpoint from {checkpoint_file}")
            return data
        except Exception as e:
            raise CheckpointError(f"Failed to load checkpoint: {e}") from e

    def list_checkpoints(self) -> list[str]:
        """List all checkpoint IDs."""
        return [f.stem for f in self.checkpoint_dir.glob("*.json")]

    def delete(self, checkpoint_id: str) -> None:
        """Delete a checkpoint."""
        checkpoint_file = self.checkpoint_dir / f"{checkpoint_id}.json"
        if checkpoint_file.exists():
            checkpoint_file.unlink()
            logger.info(f"Deleted checkpoint {checkpoint_id}")

