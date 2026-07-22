"""Normalized ground-truth annotations."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Tuple

BBox = Tuple[float, float, float, float]


@dataclass(frozen=True)
class GroundTruth:
    """One ground-truth object."""

    gt_id: str
    image_id: str
    image_index: int
    bbox_xyxy: BBox
    class_id: int
    class_name: str
    area: float
    object_size: str
    ignored: bool = False
    crowd: bool = False
    extra: Dict[str, Any] = field(default_factory=dict)

