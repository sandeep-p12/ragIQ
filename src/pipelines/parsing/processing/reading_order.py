"""Reading order determination using XY-Cut algorithm."""

import logging
from typing import List, Tuple

import numpy as np

from src.schema.document import Block, BBox

logger = logging.getLogger(__name__)


def projection_by_bboxes(boxes: np.ndarray, axis: int) -> np.ndarray:
    """
    Create projection histogram from bounding boxes.

    Args:
        boxes: Array of bounding boxes [N, 4] as (x0, y0, x1, y1)
        axis: 0 for x-axis (horizontal), 1 for y-axis (vertical)

    Returns:
        1D projection histogram
    """
    assert axis in [0, 1]
    if len(boxes) == 0:
        return np.array([])

    # Get max coordinate in projection direction
    max_coord = int(np.max(boxes[:, axis::2])) if len(boxes) > 0 else 0
    res = np.zeros(max_coord + 1, dtype=int)

    for box in boxes:
        start, end = box[axis::2]
        start_idx = max(0, int(start))
        end_idx = min(len(res), int(end))
        if start_idx < end_idx:
            res[start_idx:end_idx] += 1

    return res


def split_projection_profile(
    arr_values: np.ndarray, min_value: float, min_gap: float
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Split projection profile into groups.

    Args:
        arr_values: 1D projection array
        min_value: Minimum value to consider
        min_gap: Minimum gap between groups

    Returns:
        Tuple of (start_indices, end_indices)
    """
    arr_index = np.where(arr_values > min_value)[0]
    if len(arr_index) == 0:
        return np.array([]), np.array([])

    # Find gaps
    arr_diff = arr_index[1:] - arr_index[:-1]
    arr_diff_index = np.where(arr_diff > min_gap)[0]

    if len(arr_diff_index) == 0:
        return np.array([arr_index[0]]), np.array([arr_index[-1] + 1])

    arr_zero_intvl_start = arr_index[arr_diff_index + 1]
    arr_zero_intvl_end = arr_index[arr_diff_index]

    arr_start = np.insert(arr_zero_intvl_end, 0, arr_index[0])
    arr_end = np.append(arr_zero_intvl_start, arr_index[-1] + 1)

    return arr_start, arr_end


def recursive_xy_cut(
    boxes: np.ndarray, indices: List[int], result: List[int]
) -> None:
    """
    Recursive XY-Cut algorithm for reading order.

    Args:
        boxes: Array of bounding boxes [N, 4]
        indices: Current box indices
        result: Output list of sorted indices
    """
    if len(boxes) == 0:
        return

    # Sort by y-coordinate
    y_sorted_indices = boxes[:, 1].argsort()
    y_sorted_boxes = boxes[y_sorted_indices]
    y_sorted_orig_indices = [indices[i] for i in y_sorted_indices]

    # Project onto y-axis
    y_projection = projection_by_bboxes(y_sorted_boxes, axis=1)
    pos_y = split_projection_profile(y_projection, 0, 1)

    if len(pos_y[0]) == 0:
        return

    arr_y0, arr_y1 = pos_y

    if len(arr_y0) != len(arr_y1):
        raise ValueError(f"Mismatch: {len(arr_y0)} y0 values but {len(arr_y1)} y1 values")
    for r0, r1 in zip(arr_y0, arr_y1):
        # Filter boxes in this y-range
        mask = (r0 <= y_sorted_boxes[:, 1]) & (y_sorted_boxes[:, 1] < r1)
        chunk_boxes = y_sorted_boxes[mask]
        chunk_indices = [y_sorted_orig_indices[i] for i in range(len(mask)) if mask[i]]

        if len(chunk_boxes) == 0:
            continue

        # Sort by x-coordinate
        x_sorted_indices = chunk_boxes[:, 0].argsort()
        x_sorted_boxes = chunk_boxes[x_sorted_indices]
        x_sorted_orig_indices = [chunk_indices[i] for i in x_sorted_indices]

        # Project onto x-axis
        x_projection = projection_by_bboxes(x_sorted_boxes, axis=0)
        pos_x = split_projection_profile(x_projection, 0, 1)

        if len(pos_x[0]) == 0:
            result.extend(x_sorted_orig_indices)
            continue

        arr_x0, arr_x1 = pos_x

        if len(arr_x0) == 1:
            # Cannot split further in x-direction
            result.extend(x_sorted_orig_indices)
            continue

        # Recursively process each x-group
        if len(arr_x0) != len(arr_x1):
            raise ValueError(f"Mismatch: {len(arr_x0)} x0 values but {len(arr_x1)} x1 values")
        for c0, c1 in zip(arr_x0, arr_x1):
            mask = (c0 <= x_sorted_boxes[:, 0]) & (x_sorted_boxes[:, 0] < c1)
            sub_boxes = x_sorted_boxes[mask]
            sub_indices = [x_sorted_orig_indices[i] for i in range(len(mask)) if mask[i]]

            if len(sub_boxes) > 1:
                recursive_xy_cut(sub_boxes, sub_indices, result)
            else:
                result.extend(sub_indices)


def sort_blocks_by_reading_order(blocks: List[Block], page_width: int, page_height: int) -> List[Block]:
    """
    Sort blocks by reading order using XY-Cut algorithm.

    Args:
        blocks: List of blocks to sort
        page_width: Page width in pixels
        page_height: Page height in pixels

    Returns:
        Sorted list of blocks
    """
    if len(blocks) <= 1:
        return blocks

    # Convert blocks to bbox array
    boxes = []
    for block in blocks:
        if block.bbox is None:
            continue
        # Convert normalized bbox to absolute
        x0, y0, x1, y1 = block.bbox.to_absolute(page_width, page_height)
        boxes.append([x0, y0, x1, y1])

    if len(boxes) == 0:
        return blocks

    boxes_array = np.array(boxes)
    indices = list(range(len(blocks)))
    result_indices = []

    recursive_xy_cut(boxes_array, indices, result_indices)

    # Reorder blocks
    sorted_blocks = [blocks[i] for i in result_indices if i < len(blocks)]

    # Add any blocks that weren't processed (no bbox)
    processed_indices = set(result_indices)
    for i, block in enumerate(blocks):
        if i not in processed_indices:
            sorted_blocks.append(block)

    return sorted_blocks

