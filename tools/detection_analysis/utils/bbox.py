"""Bounding-box helpers."""

from __future__ import annotations

from typing import Iterable, Optional, Sequence, Tuple

import numpy as np

BBox = Tuple[float, float, float, float]


def to_xyxy(box: Sequence[float], fmt: str = "xyxy") -> BBox:
    """Convert a box to ``(x1, y1, x2, y2)`` floats."""

    if len(box) != 4:
        raise ValueError(f"Expected 4 bbox values, got {len(box)}.")
    x1, y1, a, b = [float(v) for v in box]
    if fmt == "xywh":
        return (x1, y1, x1 + a, y1 + b)
    if fmt != "xyxy":
        raise ValueError(f"Unsupported bbox format: {fmt}.")
    return (x1, y1, a, b)


def area(box: Sequence[float]) -> float:
    """Return non-negative box area."""

    x1, y1, x2, y2 = [float(v) for v in box]
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def object_size(box_area: float, area_ranges: dict) -> str:
    """Return the named area range containing ``box_area``."""

    for name, (low, high) in area_ranges.items():
        if box_area >= low and box_area < high:
            return name
    return "unknown"


def is_valid_box(box: Sequence[float], width: Optional[int] = None, height: Optional[int] = None) -> bool:
    """Validate box geometry and optional intersection with an image."""

    try:
        x1, y1, x2, y2 = [float(v) for v in box]
    except (TypeError, ValueError):
        return False
    if not all(np.isfinite([x1, y1, x2, y2])):
        return False
    if x2 <= x1 or y2 <= y1:
        return False
    if width is not None and height is not None:
        if x2 < 0 or y2 < 0 or x1 > width or y1 > height:
            return False
    return True


def clip_box(box: Sequence[float], width: Optional[int], height: Optional[int]) -> BBox:
    """Clip a box for display while preserving analysis coordinates elsewhere."""

    x1, y1, x2, y2 = [float(v) for v in box]
    if width is None or height is None:
        return (x1, y1, x2, y2)
    return (
        max(0.0, min(float(width), x1)),
        max(0.0, min(float(height), y1)),
        max(0.0, min(float(width), x2)),
        max(0.0, min(float(height), y2)),
    )


def iou(box_a: Sequence[float], box_b: Sequence[float]) -> float:
    """Compute intersection over union for two xyxy boxes."""

    ax1, ay1, ax2, ay2 = [float(v) for v in box_a]
    bx1, by1, bx2, by2 = [float(v) for v in box_b]
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter = area((inter_x1, inter_y1, inter_x2, inter_y2))
    union = area(box_a) + area(box_b) - inter
    if union <= 0:
        return 0.0
    return float(inter / union)


def iou_matrix(boxes_a: Iterable[Sequence[float]], boxes_b: Iterable[Sequence[float]]) -> np.ndarray:
    """Compute an ``N x M`` IoU matrix."""

    list_a = list(boxes_a)
    list_b = list(boxes_b)
    out = np.zeros((len(list_a), len(list_b)), dtype=np.float32)
    for i, a in enumerate(list_a):
        for j, b in enumerate(list_b):
            out[i, j] = iou(a, b)
    return out

