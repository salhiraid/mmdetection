"""Matching and error classification tests."""

from __future__ import annotations

from tools.detection_analysis.config import AnalysisConfig
from tools.detection_analysis.constants import (
    ERROR_BACKGROUND,
    ERROR_CLASSIFICATION,
    ERROR_CLASSIFICATION_LOCALIZATION,
    ERROR_CROWD,
    ERROR_DUPLICATE,
    ERROR_IGNORED,
    ERROR_INVALID,
    ERROR_LOCALIZATION,
    ERROR_TRUE_POSITIVE,
    GT_FALSE_NEGATIVE,
)
from tools.detection_analysis.evaluation.matcher import match_detector
from tools.detection_analysis.tests.fixtures.synthetic import dataset, pred


def _statuses(preds, ds=None):
    ds = ds or dataset()
    pred_records, gt_records = match_detector(preds, ds.ground_truths, ds.images, AnalysisConfig(confidence_threshold=0.05))
    return [p.status for p in pred_records], [g.status for g in gt_records]


def test_exact_true_positive_matching():
    pred_status, gt_status = _statuses([pred()])
    assert pred_status == [ERROR_TRUE_POSITIVE]
    assert gt_status == ["correctly_detected"]


def test_multiple_predictions_one_ground_truth_duplicate():
    pred_status, _ = _statuses([pred(idx=0, score=0.9), pred(idx=1, score=0.8)])
    assert pred_status == [ERROR_TRUE_POSITIVE, ERROR_DUPLICATE]


def test_wrong_class_high_iou_is_classification_error():
    pred_status, gt_status = _statuses([pred(label=1)])
    assert pred_status == [ERROR_CLASSIFICATION]
    assert gt_status == ["detected_with_wrong_class"]


def test_same_class_insufficient_iou_is_localization_error():
    pred_status, gt_status = _statuses([pred(box=(30.0, 30.0, 70.0, 70.0))])
    assert pred_status == [ERROR_LOCALIZATION]
    assert gt_status == ["poorly_localized"]


def test_wrong_class_poor_localization_is_classification_and_localization_error():
    pred_status, _ = _statuses([pred(box=(30.0, 30.0, 70.0, 70.0), label=1)])
    assert pred_status == [ERROR_CLASSIFICATION_LOCALIZATION]


def test_background_false_positive():
    pred_status, _ = _statuses([pred(box=(70.0, 70.0, 90.0, 90.0))])
    assert pred_status == [ERROR_BACKGROUND]


def test_false_negative():
    pred_status, gt_status = _statuses([])
    assert pred_status == []
    assert gt_status == [GT_FALSE_NEGATIVE]


def test_ignored_annotations():
    ds = dataset(ignored=True)
    pred_status, gt_status = _statuses([pred()], ds)
    assert pred_status == [ERROR_IGNORED]
    assert gt_status == ["ignored"]


def test_crowd_annotations():
    ds = dataset(crowd=True)
    pred_status, gt_status = _statuses([pred()], ds)
    assert pred_status == [ERROR_CROWD]
    assert gt_status == ["crowd_annotation"]


def test_empty_ground_truth():
    ds = dataset(with_gt=False)
    pred_status, gt_status = _statuses([pred()], ds)
    assert pred_status == [ERROR_BACKGROUND]
    assert gt_status == []


def test_invalid_boxes():
    pred_status, _ = _statuses([pred(box=(10.0, 10.0, 9.0, 50.0))])
    assert pred_status == [ERROR_INVALID]

