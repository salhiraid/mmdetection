"""Configuration objects for reproducible analysis runs."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Tuple


@dataclass
class AnalysisConfig:
    """Thresholds and switches used by matching and metrics."""

    confidence_threshold: float = 0.35
    tp_iou_threshold: float = 0.5
    localization_iou_min: float = 0.1
    duplicate_iou_threshold: float = 0.5
    classification_iou_threshold: float = 0.5
    max_detections_per_image: int = 100
    area_ranges: Dict[str, Tuple[float, float]] = field(default_factory=lambda: {
        "small": (0.0, 32.0 * 32.0),
        "medium": (32.0 * 32.0, 96.0 * 96.0),
        "large": (96.0 * 96.0, float("inf")),
    })
    class_agnostic_matching: bool = False
    ignore_crowd: bool = True

    def validate(self) -> None:
        """Raise ``ValueError`` when thresholds are inconsistent."""

        for name in (
            "confidence_threshold",
            "tp_iou_threshold",
            "localization_iou_min",
            "duplicate_iou_threshold",
            "classification_iou_threshold",
        ):
            value = getattr(self, name)
            if value < 0 or value > 1:
                raise ValueError(f"{name} must be in [0, 1], got {value}.")
        if self.localization_iou_min > self.tp_iou_threshold:
            raise ValueError("localization_iou_min cannot exceed tp_iou_threshold.")
        if self.max_detections_per_image <= 0:
            raise ValueError("max_detections_per_image must be positive.")
