"""Top-level analysis result models."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .ground_truth import GroundTruth
from .match import GroundTruthMatchRecord, MatchRecord
from .prediction import ImageRecord, Prediction


@dataclass
class AnalysisDataset:
    """Dataset records required for analysis."""

    images: List[ImageRecord]
    ground_truths: List[GroundTruth]
    classes: List[str]
    metadata: Dict[str, Any] = field(default_factory=dict)
    ann_file: Optional[str] = None


@dataclass
class DetectorAnalysis:
    """Analysis output for one detector."""

    detector_name: str
    predictions: List[Prediction]
    prediction_matches: List[MatchRecord]
    ground_truth_matches: List[GroundTruthMatchRecord]
    metrics: Dict[str, Any]
    coco_metrics: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AnalysisBundle:
    """Complete multi-detector analysis payload."""

    dataset: AnalysisDataset
    detectors: List[DetectorAnalysis]
    config: Dict[str, Any]
    comparison: Dict[str, Any] = field(default_factory=dict)
    version: str = "1.0"

