"""Compatibility wrapper for MMDetection config dataset loading."""

from __future__ import annotations

from tools.detection_analysis.config import AnalysisConfig
from tools.detection_analysis.loaders.dataset_loader import load_mmdet_config_dataset
from tools.detection_analysis.models import AnalysisDataset


def load_config_dataset(config_path: str, split: str = "test", analysis_config: AnalysisConfig | None = None) -> AnalysisDataset:
    """Load a dataset from an MMDetection config."""

    return load_mmdet_config_dataset(config_path, split=split, analysis_config=analysis_config)

