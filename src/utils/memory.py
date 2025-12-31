"""Memory management utilities for ParseForge."""

import logging
import os
from typing import Optional

logger = logging.getLogger(__name__)


def get_available_memory() -> Optional[float]:
    """
    Get available system memory in GB.

    Returns:
        Available memory in GB, or None if unable to determine
    """
    try:
        import psutil

        mem = psutil.virtual_memory()
        return mem.available / (1024**3)  # Convert to GB
    except ImportError:
        logger.warning("psutil not available, cannot determine memory")
        return None
    except Exception as e:
        logger.warning(f"Error getting memory: {e}")
        return None


def get_vram_usage() -> Optional[float]:
    """
    Get GPU VRAM usage in GB (if available).

    Returns:
        VRAM usage in GB, or None if not available
    """
    try:
        import torch

        if torch.cuda.is_available():
            return torch.cuda.memory_allocated() / (1024**3)
    except ImportError:
        pass
    except Exception as e:
        logger.debug(f"Error getting VRAM: {e}")

    return None


def calculate_batch_size(
    total_items: int,
    base_batch_size: int = 50,
    available_memory_gb: Optional[float] = None,
    item_memory_mb: float = 10.0,
) -> int:
    """
    Calculate optimal batch size based on available memory.

    Args:
        total_items: Total number of items to process
        base_batch_size: Base batch size
        available_memory_gb: Available memory in GB (if None, will try to detect)
        item_memory_mb: Estimated memory per item in MB

    Returns:
        Optimal batch size
    """
    if available_memory_gb is None:
        available_memory_gb = get_available_memory()

    if available_memory_gb is None:
        # Fallback to base batch size
        return min(base_batch_size, total_items)

    # Calculate max batch size based on memory
    # Leave 2GB for system
    usable_memory_gb = max(1.0, available_memory_gb - 2.0)
    max_batch_size = int((usable_memory_gb * 1024) / item_memory_mb)

    # Use the smaller of base, max, or total
    batch_size = min(base_batch_size, max_batch_size, total_items)

    logger.info(f"Calculated batch size: {batch_size} (available memory: {available_memory_gb:.1f}GB)")

    return max(1, batch_size)  # Ensure at least 1

