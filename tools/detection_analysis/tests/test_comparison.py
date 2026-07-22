"""Detector comparison tests."""

from __future__ import annotations

from tools.detection_analysis.config import AnalysisConfig
from tools.detection_analysis.evaluation.custom_metrics import compute_custom_metrics
from tools.detection_analysis.evaluation.detector_comparison import compare_detectors
from tools.detection_analysis.evaluation.matcher import match_detector
from tools.detection_analysis.models import DetectorAnalysis
from tools.detection_analysis.tests.fixtures.synthetic import CLASSES, dataset, pred


def test_multiple_detectors_comparison():
    ds = dataset()
    detectors = []
    for name, preds in {
        "A": [pred(detector="A")],
        "B": [pred(detector="B", box=(70, 70, 90, 90))],
    }.items():
        pred_matches, gt_matches = match_detector(preds, ds.ground_truths, ds.images, AnalysisConfig())
        detectors.append(DetectorAnalysis(name, preds, pred_matches, gt_matches, compute_custom_metrics(pred_matches, gt_matches, CLASSES, 1)))
    comparison = compare_detectors(detectors)
    assert comparison["metrics_table"][0]["detector"] == "A"
    assert comparison["rankings"]["f1"][0]["detector"] == "A"
    assert comparison["image_states"]["1"]["state"] == "correct_only_for_A"

