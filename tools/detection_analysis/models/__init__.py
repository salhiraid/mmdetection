"""Typed data models used by the detection analysis tool."""

from .analysis_result import AnalysisBundle, AnalysisDataset, DetectorAnalysis
from .ground_truth import GroundTruth
from .match import GroundTruthMatchRecord, MatchRecord
from .prediction import ImageRecord, Prediction

__all__ = [
    "AnalysisBundle",
    "AnalysisDataset",
    "DetectorAnalysis",
    "GroundTruth",
    "GroundTruthMatchRecord",
    "ImageRecord",
    "MatchRecord",
    "Prediction",
]

