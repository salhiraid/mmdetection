"""Prediction and ground-truth status records."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

BBox = Tuple[float, float, float, float]


@dataclass(frozen=True)
class MatchRecord:
    """Primary status assigned to a prediction."""

    detector_name: str
    image_id: str
    image_index: int
    prediction_id: str
    status: str
    score: float
    pred_class_id: int
    pred_class_name: str
    pred_bbox: BBox
    iou: float = 0.0
    matched_gt_id: Optional[str] = None
    gt_class_id: Optional[int] = None
    gt_class_name: Optional[str] = None
    gt_bbox: Optional[BBox] = None
    object_size: Optional[str] = None
    valid: bool = True


@dataclass(frozen=True)
class GroundTruthMatchRecord:
    """Primary status assigned to a ground-truth object."""

    detector_name: str
    image_id: str
    image_index: int
    gt_id: str
    status: str
    gt_class_id: int
    gt_class_name: str
    gt_bbox: BBox
    object_size: str
    matched_prediction_id: Optional[str] = None
    pred_class_id: Optional[int] = None
    pred_class_name: Optional[str] = None
    score: Optional[float] = None
    iou: float = 0.0

