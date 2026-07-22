"""Small synthetic detection fixtures."""

from __future__ import annotations

from tools.detection_analysis.config import AnalysisConfig
from tools.detection_analysis.models import AnalysisDataset, GroundTruth, ImageRecord, Prediction
from tools.detection_analysis.utils.bbox import area, object_size


CLASSES = ["cat", "dog"]


def dataset(with_gt: bool = True, ignored: bool = False, crowd: bool = False) -> AnalysisDataset:
    """Return a one-image dataset with one optional object."""

    cfg = AnalysisConfig()
    images = [ImageRecord(0, "1", "img.jpg", "img.jpg", 100, 100)]
    gts = []
    if with_gt:
        box = (10.0, 10.0, 50.0, 50.0)
        gts.append(GroundTruth(
            gt_id="1:0",
            image_id="1",
            image_index=0,
            bbox_xyxy=box,
            class_id=0,
            class_name="cat",
            area=area(box),
            object_size=object_size(area(box), cfg.area_ranges),
            ignored=ignored,
            crowd=crowd,
        ))
    return AnalysisDataset(images=images, ground_truths=gts, classes=CLASSES)


def pred(box=(10.0, 10.0, 50.0, 50.0), label=0, score=0.9, detector="A", idx=0) -> Prediction:
    """Create a prediction on the synthetic image."""

    return Prediction(
        prediction_id=f"{detector}:1:{idx}",
        image_id="1",
        image_index=0,
        bbox_xyxy=box,
        score=score,
        class_id=label,
        class_name=CLASSES[label] if 0 <= label < len(CLASSES) else str(label),
        detector_name=detector,
        source_index=idx,
    )

