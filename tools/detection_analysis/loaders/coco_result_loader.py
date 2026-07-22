"""Compatibility wrapper for COCO JSON prediction files."""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

from tools.detection_analysis.loaders.result_adapter import normalize_result_file
from tools.detection_analysis.models import AnalysisDataset, Prediction


def load_coco_results(path: str | Path, dataset: AnalysisDataset, detector_name: Optional[str] = None) -> List[Prediction]:
    """Load and normalize a COCO JSON prediction file."""

    return normalize_result_file(path, dataset, detector_name)

