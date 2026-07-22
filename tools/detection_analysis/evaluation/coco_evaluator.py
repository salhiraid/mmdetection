"""COCO metric integration."""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any, Dict, List

from tools.detection_analysis.models import AnalysisDataset, Prediction
from tools.detection_analysis.utils.serialization import dump_json


def evaluate_coco(dataset: AnalysisDataset, predictions: List[Prediction]) -> Dict[str, Any]:
    """Evaluate bbox predictions with pycocotools when it is available."""

    try:
        from pycocotools.coco import COCO
        from pycocotools.cocoeval import COCOeval
    except Exception as exc:
        return {"available": False, "reason": f"pycocotools unavailable: {exc}"}
    if not dataset.images:
        return {"available": False, "reason": "empty dataset"}

    with tempfile.TemporaryDirectory() as tmp:
        gt_path = Path(tmp) / "gt.json"
        pred_path = Path(tmp) / "pred.json"
        dump_json(gt_path, _dataset_to_coco(dataset))
        dump_json(pred_path, _predictions_to_coco(predictions))
        coco_gt = COCO(str(gt_path))
        if not predictions:
            return _empty_metrics()
        coco_dt = coco_gt.loadRes(str(pred_path))
        coco_eval = COCOeval(coco_gt, coco_dt, "bbox")
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
        stats = coco_eval.stats
        metrics = {
            "mAP": float(stats[0]),
            "AP50": float(stats[1]),
            "AP75": float(stats[2]),
            "AP_small": float(stats[3]),
            "AP_medium": float(stats[4]),
            "AP_large": float(stats[5]),
            "AR@1": float(stats[6]),
            "AR@10": float(stats[7]),
            "AR@100": float(stats[8]),
            "AR_small": float(stats[9]),
            "AR_medium": float(stats[10]),
            "AR_large": float(stats[11]),
            "per_class_AP": _per_class_ap(coco_eval, dataset.classes),
        }
        return metrics


def _dataset_to_coco(dataset: AnalysisDataset) -> Dict[str, Any]:
    categories = [{"id": i, "name": name} for i, name in enumerate(dataset.classes)]
    images = [
        {
            "id": int(img.image_id) if str(img.image_id).isdigit() else img.image_index,
            "file_name": img.file_name,
            "width": img.width or 0,
            "height": img.height or 0,
        }
        for img in dataset.images
    ]
    image_id_to_coco = {img.image_id: images[i]["id"] for i, img in enumerate(dataset.images)}
    annotations = []
    for idx, gt in enumerate(dataset.ground_truths):
        x1, y1, x2, y2 = gt.bbox_xyxy
        annotations.append({
            "id": idx + 1,
            "image_id": image_id_to_coco[gt.image_id],
            "category_id": gt.class_id,
            "bbox": [x1, y1, x2 - x1, y2 - y1],
            "area": gt.area,
            "iscrowd": 1 if gt.crowd else 0,
            "ignore": 1 if gt.ignored else 0,
        })
    return {"images": images, "annotations": annotations, "categories": categories}


def _predictions_to_coco(predictions: List[Prediction]) -> List[Dict[str, Any]]:
    out = []
    for pred in predictions:
        x1, y1, x2, y2 = pred.bbox_xyxy
        image_id = int(pred.image_id) if str(pred.image_id).isdigit() else pred.image_index
        out.append({
            "image_id": image_id,
            "category_id": pred.class_id,
            "bbox": [x1, y1, x2 - x1, y2 - y1],
            "score": pred.score,
        })
    return out


def _empty_metrics() -> Dict[str, Any]:
    return {
        "mAP": 0.0,
        "AP50": 0.0,
        "AP75": 0.0,
        "AP_small": 0.0,
        "AP_medium": 0.0,
        "AP_large": 0.0,
        "AR@1": 0.0,
        "AR@10": 0.0,
        "AR@100": 0.0,
        "AR_small": 0.0,
        "AR_medium": 0.0,
        "AR_large": 0.0,
        "per_class_AP": {},
    }


def _per_class_ap(coco_eval: Any, class_names: List[str]) -> Dict[str, float]:
    precisions = coco_eval.eval.get("precision")
    if precisions is None:
        return {}
    out = {}
    for idx, name in enumerate(class_names):
        values = precisions[:, :, idx, 0, -1]
        values = values[values > -1]
        out[name] = float(values.mean()) if values.size else 0.0
    return out

