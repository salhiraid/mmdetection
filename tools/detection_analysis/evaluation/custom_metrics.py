"""Custom detection metrics derived from match records."""

from __future__ import annotations

from collections import Counter, defaultdict
from statistics import median
from typing import Any, Dict, Iterable, List

from tools.detection_analysis.constants import (
    ERROR_BACKGROUND,
    ERROR_CLASSIFICATION,
    ERROR_CLASSIFICATION_LOCALIZATION,
    ERROR_DUPLICATE,
    ERROR_INVALID,
    ERROR_LOCALIZATION,
    ERROR_TRUE_POSITIVE,
    GT_DUPLICATE_ONLY,
    GT_FALSE_NEGATIVE,
    GT_POOR_LOCALIZATION,
    GT_WRONG_CLASS,
)
from tools.detection_analysis.models import GroundTruthMatchRecord, MatchRecord

FP_STATUSES = {
    ERROR_BACKGROUND,
    ERROR_CLASSIFICATION,
    ERROR_LOCALIZATION,
    ERROR_CLASSIFICATION_LOCALIZATION,
    ERROR_DUPLICATE,
    ERROR_INVALID,
}

FN_GT_STATUSES = {
    GT_FALSE_NEGATIVE,
    GT_WRONG_CLASS,
    GT_POOR_LOCALIZATION,
    GT_DUPLICATE_ONLY,
}


def compute_custom_metrics(
    prediction_matches: Iterable[MatchRecord],
    ground_truth_matches: Iterable[GroundTruthMatchRecord],
    class_names: List[str],
    num_images: int,
) -> Dict[str, Any]:
    """Compute global, per-class, and per-image metrics."""

    preds = list(prediction_matches)
    gts = list(ground_truth_matches)
    status_counts = Counter(p.status for p in preds)
    gt_status_counts = Counter(g.status for g in gts)
    tp = status_counts[ERROR_TRUE_POSITIVE]
    fp = sum(status_counts[s] for s in FP_STATUSES)
    fn = sum(gt_status_counts[s] for s in FN_GT_STATUSES)
    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    f1 = _safe_div(2 * precision * recall, precision + recall)
    tp_ious = [p.iou for p in preds if p.status == ERROR_TRUE_POSITIVE]

    per_class: Dict[str, Dict[str, Any]] = {}
    for class_id, class_name in enumerate(class_names):
        class_tp = sum(1 for p in preds if p.status == ERROR_TRUE_POSITIVE and p.pred_class_id == class_id)
        class_fp = sum(1 for p in preds if p.pred_class_id == class_id and p.status in FP_STATUSES)
        class_fn = sum(1 for g in gts if g.gt_class_id == class_id and g.status in FN_GT_STATUSES)
        cp = _safe_div(class_tp, class_tp + class_fp)
        cr = _safe_div(class_tp, class_tp + class_fn)
        per_class[class_name] = {
            "class_id": class_id,
            "tp": class_tp,
            "fp": class_fp,
            "fn": class_fn,
            "precision": cp,
            "recall": cr,
            "f1": _safe_div(2 * cp * cr, cp + cr),
        }

    per_image = _per_image_summary(preds, gts)
    images_with_errors = sum(1 for row in per_image.values() if row["errors"] > 0)
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "prediction_status_counts": dict(status_counts),
        "ground_truth_status_counts": dict(gt_status_counts),
        "duplicate_detections": status_counts[ERROR_DUPLICATE],
        "classification_errors": status_counts[ERROR_CLASSIFICATION],
        "localization_errors": status_counts[ERROR_LOCALIZATION],
        "classification_localization_errors": status_counts[ERROR_CLASSIFICATION_LOCALIZATION],
        "background_false_positives": status_counts[ERROR_BACKGROUND],
        "mean_tp_iou": sum(tp_ious) / len(tp_ious) if tp_ious else 0.0,
        "median_tp_iou": median(tp_ious) if tp_ious else 0.0,
        "confidence_distribution": _confidence_distribution(preds),
        "error_rate_by_object_size": _error_rate_by_object_size(preds, gts),
        "error_rate_by_class": {
            name: 1.0 - values["f1"] for name, values in per_class.items()
        },
        "error_rate_by_image": {image_id: row["errors"] for image_id, row in per_image.items()},
        "images_with_no_errors": max(0, num_images - images_with_errors),
        "images_with_errors": images_with_errors,
        "average_predictions_per_image": _safe_div(len(preds), num_images),
        "average_ground_truth_objects_per_image": _safe_div(len(gts), num_images),
        "per_class": per_class,
        "per_image": per_image,
    }


def _safe_div(num: float, den: float) -> float:
    return float(num / den) if den else 0.0


def _confidence_distribution(preds: List[MatchRecord]) -> Dict[str, List[float]]:
    by_status: Dict[str, List[float]] = defaultdict(list)
    for pred in preds:
        by_status[pred.status].append(pred.score)
    by_status["all"] = [p.score for p in preds]
    return dict(by_status)


def _per_image_summary(preds: List[MatchRecord], gts: List[GroundTruthMatchRecord]) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0, "errors": 0, "predictions": 0, "ground_truth": 0})
    for pred in preds:
        row = out[pred.image_id]
        row["predictions"] += 1
        if pred.status == ERROR_TRUE_POSITIVE:
            row["tp"] += 1
        elif pred.status in FP_STATUSES:
            row["fp"] += 1
            row["errors"] += 1
    for gt in gts:
        row = out[gt.image_id]
        row["ground_truth"] += 1
        if gt.status in FN_GT_STATUSES:
            row["fn"] += 1
            row["errors"] += 1
    return dict(out)


def _error_rate_by_object_size(preds: List[MatchRecord], gts: List[GroundTruthMatchRecord]) -> Dict[str, float]:
    totals: Counter[str] = Counter()
    errors: Counter[str] = Counter()
    for gt in gts:
        totals[gt.object_size] += 1
        if gt.status in FN_GT_STATUSES:
            errors[gt.object_size] += 1
    for pred in preds:
        if pred.object_size:
            totals[pred.object_size] += 1
            if pred.status in FP_STATUSES:
                errors[pred.object_size] += 1
    return {name: _safe_div(errors[name], totals[name]) for name in sorted(totals)}
