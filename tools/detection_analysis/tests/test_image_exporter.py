"""Tests for full-resolution visualization exports."""

from PIL import Image

from tools.detection_analysis.export.image_exporter import export_bundle_images
from tools.detection_analysis.models import (
    AnalysisBundle,
    AnalysisDataset,
    DetectorAnalysis,
    GroundTruthMatchRecord,
    ImageRecord,
    MatchRecord,
)


def test_export_bundle_images_preserves_resolution_and_filters_confidence(tmp_path):
    source = tmp_path / "source.jpg"
    Image.new("RGB", (321, 123), "white").save(source)
    image = ImageRecord(0, "image-1", "source.jpg", str(source), 321, 123)
    matches = [
        _match("high", "background_false_positive", 0.35, (10, 10, 30, 30)),
        _match("low", "background_false_positive", 0.34, (40, 10, 60, 30)),
        _match("correct", "true_positive", 0.9, (70, 10, 90, 30)),
    ]
    false_negative = GroundTruthMatchRecord(
        detector_name="detector",
        image_id="image-1",
        image_index=0,
        gt_id="gt-1",
        status="false_negative",
        gt_class_id=0,
        gt_class_name="cat",
        gt_bbox=(100, 10, 120, 30),
        object_size="small",
    )
    detector = DetectorAnalysis(
        "detector", [], matches, [false_negative], metrics={}
    )
    bundle = AnalysisBundle(
        AnalysisDataset([image], [], ["cat"]),
        [detector],
        {"confidence_threshold": 0.05},
    )

    export_bundle_images(bundle, tmp_path / "visualizations")

    predictions = tmp_path / "visualizations/detector/predictions_only/000000_source.jpg"
    problems = tmp_path / "visualizations/detector/problems_only/000000_source.jpg"
    assert predictions.exists()
    assert problems.exists()
    with Image.open(predictions) as exported_predictions:
        assert exported_predictions.size == (321, 123)
    with Image.open(problems) as exported_problems:
        assert exported_problems.size == (321, 123)


def test_export_bundle_images_skips_clean_problem_image(tmp_path):
    source = tmp_path / "clean.png"
    Image.new("RGB", (80, 60), "white").save(source)
    image = ImageRecord(0, "image-1", "clean.png", str(source), 80, 60)
    detector = DetectorAnalysis(
        "detector", [],
        [_match("correct", "true_positive", 0.9, (1, 1, 20, 20))],
        [],
        metrics={})
    bundle = AnalysisBundle(
        AnalysisDataset([image], [], ["cat"]),
        [detector],
        {"confidence_threshold": 0.35},
    )

    export_bundle_images(bundle, tmp_path / "visualizations")

    predictions = tmp_path / "visualizations/detector/predictions_only/000000_clean.png"
    problems = tmp_path / "visualizations/detector/problems_only/000000_clean.png"
    assert predictions.exists()
    assert not problems.exists()


def _match(prediction_id, status, score, bbox):
    return MatchRecord(
        detector_name="detector",
        image_id="image-1",
        image_index=0,
        prediction_id=prediction_id,
        status=status,
        score=score,
        pred_class_id=0,
        pred_class_name="cat",
        pred_bbox=bbox,
    )
