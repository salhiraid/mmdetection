"""Normalized prediction and image records."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

BBox = Tuple[float, float, float, float]


@dataclass(frozen=True)
class ImageRecord:
    """Dataset image metadata in analysis order."""

    image_index: int
    image_id: str
    file_name: str
    img_path: str
    width: Optional[int] = None
    height: Optional[int] = None


@dataclass(frozen=True)
class Prediction:
    """One normalized detector prediction."""

    prediction_id: str
    image_id: str
    image_index: int
    bbox_xyxy: BBox
    score: float
    class_id: int
    class_name: str
    detector_name: str
    source_index: int = 0
    extra: Dict[str, Any] = field(default_factory=dict)

