"""Custom metric tests."""

from __future__ import annotations

from tools.detection_analysis.config import AnalysisConfig
from tools.detection_analysis.evaluation.confusion_matrix import build_confusion_matrix
from tools.detection_analysis.evaluation.custom_metrics import compute_custom_metrics
from tools.detection_analysis.evaluation.matcher import match_detector
from tools.detection_analysis.tests.fixtures.synthetic import CLASSES, dataset, pred


def test_per_class_metrics_and_confusion_matrix():
    ds = dataset()
    pred_matches, gt_matches = match_detector([pred(), pred(box=(70, 70, 90, 90), idx=1)], ds.ground_truths, ds.images, AnalysisConfig())
    metrics = compute_custom_metrics(pred_matches, gt_matches, CLASSES, 1)
    assert metrics["tp"] == 1
    assert metrics["fp"] == 1
    assert metrics["fn"] == 0
    assert metrics["per_class"]["cat"]["precision"] == 0.5
    matrix = build_confusion_matrix(pred_matches, gt_matches, CLASSES)
    assert matrix.loc["cat", "cat"] == 1
    assert matrix.loc["background", "cat"] == 1

