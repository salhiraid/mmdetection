"""Result normalization tests."""

from __future__ import annotations

import numpy as np

from tools.detection_analysis.loaders.result_adapter import normalize_results
from tools.detection_analysis.tests.fixtures.synthetic import dataset


def test_dict_result_format_conversion():
    ds = dataset()
    raw = [{"pred_instances": {"bboxes": np.array([[10, 10, 50, 50]]), "scores": np.array([0.9]), "labels": np.array([0])}}]
    preds = normalize_results(raw, ds, "A")
    assert len(preds) == 1
    assert preds[0].class_name == "cat"


def test_old_mmdetection_per_class_conversion():
    ds = dataset()
    raw = [[np.array([[10, 10, 50, 50, 0.9]]), np.zeros((0, 5))]]
    preds = normalize_results(raw, ds, "A")
    assert len(preds) == 1
    assert preds[0].score == 0.9


def test_coco_json_conversion():
    ds = dataset()
    ds.metadata["categories"] = [{"id": 1, "name": "cat"}, {"id": 2, "name": "dog"}]
    preds = normalize_results([{"image_id": 1, "bbox": [10, 10, 40, 40], "score": 0.7, "category_id": 1}], ds, "A")
    assert len(preds) == 1
    assert preds[0].bbox_xyxy == (10.0, 10.0, 50.0, 50.0)

