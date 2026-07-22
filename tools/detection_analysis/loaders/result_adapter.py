"""Normalize MMDetection and COCO result formats."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import numpy as np

from tools.detection_analysis.models import AnalysisDataset, Prediction
from tools.detection_analysis.utils.bbox import to_xyxy
from tools.detection_analysis.utils.validation import ResultFormatError


def load_result_file(path: str | Path) -> Any:
    """Load a result file using MMEngine when available, falling back to pickle/json."""

    path = Path(path)
    if not path.exists():
        raise ResultFormatError(f"Result file does not exist: {path}")
    if path.suffix.lower() == ".json":
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    try:
        from mmengine.fileio import load
        return load(str(path))
    except Exception:
        import pickle
        with path.open("rb") as f:
            return pickle.load(f)


def normalize_result_file(path: str | Path, dataset: AnalysisDataset, detector_name: Optional[str] = None) -> List[Prediction]:
    """Load and normalize one result file."""

    name = detector_name or Path(path).stem
    return normalize_results(load_result_file(path), dataset=dataset, detector_name=name)


def normalize_results(raw: Any, dataset: AnalysisDataset, detector_name: str) -> List[Prediction]:
    """Convert supported detector outputs to ``Prediction`` records."""

    if isinstance(raw, dict) and "predictions" in raw:
        raw = raw["predictions"]
    if isinstance(raw, list) and _looks_like_coco_json_predictions(raw):
        return _from_coco_predictions(raw, dataset, detector_name)
    if not isinstance(raw, (list, tuple)):
        raise ResultFormatError(
            "Unsupported result format. Expected a list/tuple, COCO JSON list, "
            "or a dict containing 'predictions'.")
    if len(raw) != len(dataset.images):
        raise ResultFormatError(
            f"Prediction image count ({len(raw)}) does not match dataset image count ({len(dataset.images)}).")
    predictions: List[Prediction] = []
    for image_index, per_image in enumerate(raw):
        image = dataset.images[image_index]
        predictions.extend(_normalize_per_image(per_image, image.image_id, image_index, detector_name, dataset.classes))
    return predictions


def _looks_like_coco_json_predictions(raw: Sequence[Any]) -> bool:
    if not raw:
        return False
    first = raw[0]
    return isinstance(first, dict) and {"image_id", "bbox", "score", "category_id"}.issubset(first.keys())


def _from_coco_predictions(raw: Sequence[Dict[str, Any]], dataset: AnalysisDataset, detector_name: str) -> List[Prediction]:
    image_id_to_index = {img.image_id: img.image_index for img in dataset.images}
    cat_id_to_label = {
        int(cat.get("id")): i
        for i, cat in enumerate(dataset.metadata.get("categories", []))
        if isinstance(cat, dict) and "id" in cat
    }
    predictions = []
    for idx, item in enumerate(raw):
        image_id = str(item["image_id"])
        if image_id not in image_id_to_index:
            raise ResultFormatError(f"COCO prediction references unknown image_id {image_id}.")
        label = cat_id_to_label.get(int(item.get("category_id", 0)), int(item.get("category_id", 0)))
        class_name = dataset.classes[label] if 0 <= label < len(dataset.classes) else str(label)
        predictions.append(Prediction(
            prediction_id=f"{detector_name}:{image_id}:{idx}",
            image_id=image_id,
            image_index=image_id_to_index[image_id],
            bbox_xyxy=to_xyxy(item["bbox"], fmt="xywh"),
            score=float(item.get("score", 0.0)),
            class_id=label,
            class_name=class_name,
            detector_name=detector_name,
            source_index=idx,
            extra={"category_id": item.get("category_id")},
        ))
    return predictions


def _normalize_per_image(
    per_image: Any,
    image_id: str,
    image_index: int,
    detector_name: str,
    classes: Sequence[str],
) -> List[Prediction]:
    if isinstance(per_image, tuple):
        per_image = per_image[0]
    if hasattr(per_image, "pred_instances"):
        per_image = {"pred_instances": per_image.pred_instances}
    if isinstance(per_image, dict) and "pred_instances" in per_image:
        return _from_instance_container(per_image["pred_instances"], image_id, image_index, detector_name, classes)
    if _is_instance_container(per_image):
        return _from_instance_container(per_image, image_id, image_index, detector_name, classes)
    if isinstance(per_image, dict) and {"bboxes", "scores", "labels"}.issubset(per_image.keys()):
        return _from_arrays(per_image["bboxes"], per_image["scores"], per_image["labels"], image_id, image_index, detector_name, classes)
    if isinstance(per_image, (list, tuple)):
        return _from_old_per_class(per_image, image_id, image_index, detector_name, classes)
    raise ResultFormatError(f"Unsupported per-image result at dataset index {image_index}: {type(per_image)!r}")


def _is_instance_container(value: Any) -> bool:
    return all(hasattr(value, key) or (isinstance(value, dict) and key in value) for key in ("bboxes", "scores", "labels"))


def _get(container: Any, key: str) -> Any:
    if isinstance(container, dict):
        return container[key]
    return getattr(container, key)


def _as_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    elif hasattr(value, "cpu") and hasattr(value.cpu(), "numpy"):
        value = value.cpu().numpy()
    return np.asarray(value)


def _from_instance_container(container: Any, image_id: str, image_index: int, detector_name: str, classes: Sequence[str]) -> List[Prediction]:
    return _from_arrays(
        _get(container, "bboxes"),
        _get(container, "scores"),
        _get(container, "labels"),
        image_id,
        image_index,
        detector_name,
        classes,
    )


def _from_arrays(
    bboxes: Any,
    scores: Any,
    labels: Any,
    image_id: str,
    image_index: int,
    detector_name: str,
    classes: Sequence[str],
) -> List[Prediction]:
    boxes = _as_numpy(bboxes)
    scores_arr = _as_numpy(scores).reshape(-1)
    labels_arr = _as_numpy(labels).reshape(-1)
    if boxes.size == 0:
        return []
    boxes = boxes.reshape((-1, boxes.shape[-1]))
    if boxes.shape[1] == 5 and len(scores_arr) != boxes.shape[0]:
        scores_arr = boxes[:, 4]
        boxes = boxes[:, :4]
    if boxes.shape[1] != 4:
        raise ResultFormatError(f"Expected bboxes shape Nx4 or Nx5, got {boxes.shape}.")
    if len(scores_arr) != len(boxes) or len(labels_arr) != len(boxes):
        raise ResultFormatError("bboxes, scores, and labels must have the same length.")
    out: List[Prediction] = []
    for i, (box, score, label) in enumerate(zip(boxes, scores_arr, labels_arr)):
        class_id = int(label)
        class_name = classes[class_id] if 0 <= class_id < len(classes) else str(class_id)
        out.append(Prediction(
            prediction_id=f"{detector_name}:{image_id}:{i}",
            image_id=str(image_id),
            image_index=image_index,
            bbox_xyxy=to_xyxy(box.tolist(), fmt="xyxy"),
            score=float(score),
            class_id=class_id,
            class_name=class_name,
            detector_name=detector_name,
            source_index=i,
        ))
    return out


def _from_old_per_class(per_class: Sequence[Any], image_id: str, image_index: int, detector_name: str, classes: Sequence[str]) -> List[Prediction]:
    out: List[Prediction] = []
    source_index = 0
    for class_id, boxes_for_class in enumerate(per_class):
        boxes = _as_numpy(boxes_for_class)
        if boxes.size == 0:
            continue
        boxes = boxes.reshape((-1, boxes.shape[-1]))
        if boxes.shape[1] < 5:
            raise ResultFormatError("Old MMDetection per-class arrays must be Nx5 with score in the fifth column.")
        class_name = classes[class_id] if 0 <= class_id < len(classes) else str(class_id)
        for box in boxes:
            out.append(Prediction(
                prediction_id=f"{detector_name}:{image_id}:{source_index}",
                image_id=str(image_id),
                image_index=image_index,
                bbox_xyxy=to_xyxy(box[:4].tolist(), fmt="xyxy"),
                score=float(box[4]),
                class_id=class_id,
                class_name=class_name,
                detector_name=detector_name,
                source_index=source_index,
            ))
            source_index += 1
    return out

