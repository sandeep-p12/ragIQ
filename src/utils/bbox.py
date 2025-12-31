"""Bounding box utilities for ParseForge."""

import math
from typing import Tuple


def calculate_iou(bbox1: Tuple[float, float, float, float], bbox2: Tuple[float, float, float, float]) -> float:
    """
    Calculate Intersection over Union (IoU) of two bounding boxes.

    Args:
        bbox1: First bounding box as (x0, y0, x1, y1)
        bbox2: Second bounding box as (x0, y0, x1, y1)

    Returns:
        IoU value between 0 and 1
    """
    x1_min, y1_min, x1_max, y1_max = bbox1
    x2_min, y2_min, x2_max, y2_max = bbox2

    # Calculate intersection
    x_left = max(x1_min, x2_min)
    y_top = max(y1_min, y2_min)
    x_right = min(x1_max, x2_max)
    y_bottom = min(y1_max, y2_max)

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    # Calculate areas
    bbox1_area = (x1_max - x1_min) * (y1_max - y1_min)
    bbox2_area = (x2_max - x2_min) * (y2_max - y2_min)

    if bbox1_area == 0 or bbox2_area == 0:
        return 0.0

    # Calculate union
    union_area = bbox1_area + bbox2_area - intersection_area

    return intersection_area / union_area if union_area > 0 else 0.0


def bbox_distance(bbox1: Tuple[float, float, float, float], bbox2: Tuple[float, float, float, float]) -> float:
    """
    Calculate distance between two bounding boxes.

    Args:
        bbox1: First bounding box as (x0, y0, x1, y1)
        bbox2: Second bounding box as (x0, y0, x1, y1)

    Returns:
        Distance between boxes (0 if overlapping)
    """
    def dist(point1: Tuple[float, float], point2: Tuple[float, float]) -> float:
        return math.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)

    x1, y1, x1b, y1b = bbox1
    x2, y2, x2b, y2b = bbox2

    # Check relative positions
    left = x2b < x1
    right = x1b < x2
    bottom = y2b < y1
    top = y1b < y2

    if top and left:
        return dist((x1, y1b), (x2b, y2))
    elif left and bottom:
        return dist((x1, y1), (x2b, y2b))
    elif bottom and right:
        return dist((x1b, y1), (x2, y2b))
    elif right and top:
        return dist((x1b, y1b), (x2, y2))
    elif left:
        return x1 - x2b
    elif right:
        return x2 - x1b
    elif bottom:
        return y1 - y2b
    elif top:
        return y2 - y1b
    return 0.0


def is_overlapping(bbox1: Tuple[float, float, float, float], bbox2: Tuple[float, float, float, float]) -> bool:
    """
    Check if two bounding boxes overlap.

    Args:
        bbox1: First bounding box as (x0, y0, x1, y1)
        bbox2: Second bounding box as (x0, y0, x1, y1)

    Returns:
        True if boxes overlap, False otherwise
    """
    x1_min, y1_min, x1_max, y1_max = bbox1
    x2_min, y2_min, x2_max, y2_max = bbox2

    return not (x1_max < x2_min or x2_max < x1_min or y1_max < y2_min or y2_max < y1_min)


def is_contained(bbox1: Tuple[float, float, float, float], bbox2: Tuple[float, float, float, float]) -> bool:
    """
    Check if bbox1 is completely contained within bbox2.

    Args:
        bbox1: Inner bounding box as (x0, y0, x1, y1)
        bbox2: Outer bounding box as (x0, y0, x1, y1)

    Returns:
        True if bbox1 is contained in bbox2
    """
    x1_min, y1_min, x1_max, y1_max = bbox1
    x2_min, y2_min, x2_max, y2_max = bbox2

    return (
        x1_min >= x2_min
        and y1_min >= y2_min
        and x1_max <= x2_max
        and y1_max <= y2_max
    )


def calculate_overlap_ratio(bbox1: Tuple[float, float, float, float], bbox2: Tuple[float, float, float, float]) -> float:
    """
    Calculate overlap area ratio relative to the smaller box.

    Args:
        bbox1: First bounding box as (x0, y0, x1, y1)
        bbox2: Second bounding box as (x0, y0, x1, y1)

    Returns:
        Overlap ratio (0-1)
    """
    x1_min, y1_min, x1_max, y1_max = bbox1
    x2_min, y2_min, x2_max, y2_max = bbox2

    # Calculate intersection
    x_left = max(x1_min, x2_min)
    y_top = max(y1_min, y2_min)
    x_right = min(x1_max, x2_max)
    y_bottom = min(y1_max, y2_max)

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    # Calculate areas
    area1 = (x1_max - x1_min) * (y1_max - y1_min)
    area2 = (x2_max - x2_min) * (y2_max - y2_min)
    min_area = min(area1, area2)

    if min_area == 0:
        return 0.0

    return intersection_area / min_area


def normalize_bbox(bbox: Tuple[float, float, float, float], width: int, height: int) -> Tuple[float, float, float, float]:
    """
    Normalize bounding box coordinates to 0-1 range.

    Args:
        bbox: Bounding box in absolute coordinates (x0, y0, x1, y1)
        width: Image/page width
        height: Image/page height

    Returns:
        Normalized bounding box (x0, y0, x1, y1) in 0-1 range
    """
    x0, y0, x1, y1 = bbox
    if width == 0 or height == 0:
        return (0.0, 0.0, 1.0, 1.0)
    return (x0 / width, y0 / height, x1 / width, y1 / height)


def denormalize_bbox(bbox: Tuple[float, float, float, float], width: int, height: int) -> Tuple[int, int, int, int]:
    """
    Convert normalized bounding box to absolute coordinates.

    Args:
        bbox: Normalized bounding box (x0, y0, x1, y1) in 0-1 range
        width: Image/page width
        height: Image/page height

    Returns:
        Absolute bounding box coordinates (x0, y0, x1, y1)
    """
    x0, y0, x1, y1 = bbox
    return (int(x0 * width), int(y0 * height), int(x1 * width), int(y1 * height))

