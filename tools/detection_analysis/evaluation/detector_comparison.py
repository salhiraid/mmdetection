"""Multi-detector comparison helpers."""

from __future__ import annotations

from typing import Any, Dict, List

from tools.detection_analysis.constants import ERROR_TRUE_POSITIVE, GT_FALSE_NEGATIVE
from tools.detection_analysis.models import DetectorAnalysis


HIGHER_IS_BETTER = {
    "mAP": True,
    "AP50": True,
    "AP75": True,
    "precision": True,
    "recall": True,
    "f1": True,
    "mean_tp_iou": True,
    "duplicate_rate": False,
    "background_fp_rate": False,
}


def compare_detectors(detectors: List[DetectorAnalysis]) -> Dict[str, Any]:
    """Create ranking tables and per-image disagreement summaries."""

    rows = []
    for det in detectors:
        metrics = det.metrics
        fp = max(metrics.get("fp", 0), 1)
        row = {
            "detector": det.detector_name,
            "precision": metrics.get("precision", 0.0),
            "recall": metrics.get("recall", 0.0),
            "f1": metrics.get("f1", 0.0),
            "mean_tp_iou": metrics.get("mean_tp_iou", 0.0),
            "duplicate_rate": metrics.get("duplicate_detections", 0) / fp,
            "background_fp_rate": metrics.get("background_false_positives", 0) / fp,
        }
        for key, value in det.coco_metrics.items():
            row[key] = value
        rows.append(row)

    rankings = {}
    for metric, higher in HIGHER_IS_BETTER.items():
        present = [row for row in rows if metric in row]
        rankings[metric] = sorted(present, key=lambda r: r[metric], reverse=higher)

    return {
        "metrics_table": rows,
        "rankings": rankings,
        "metric_differences": _metric_differences(rows),
        "image_states": _image_states(detectors),
        "higher_is_better": HIGHER_IS_BETTER,
    }


def _metric_differences(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not rows:
        return []
    ref = rows[0]
    out = []
    for row in rows[1:]:
        diff = {"detector": row["detector"], "reference": ref["detector"]}
        for key, value in row.items():
            if key != "detector" and isinstance(value, (int, float)) and isinstance(ref.get(key), (int, float)):
                diff[key] = value - ref[key]
        out.append(diff)
    return out


def _image_states(detectors: List[DetectorAnalysis]) -> Dict[str, Dict[str, Any]]:
    by_image: Dict[str, Dict[str, Any]] = {}
    for det in detectors:
        per_image = det.metrics.get("per_image", {})
        for image_id, row in per_image.items():
            by_image.setdefault(image_id, {})[det.detector_name] = row
    states = {}
    for image_id, values in by_image.items():
        correct = {name: row.get("errors", 0) == 0 for name, row in values.items()}
        fn = {name: row.get("fn", 0) for name, row in values.items()}
        fp = {name: row.get("fp", 0) for name, row in values.items()}
        if correct and all(correct.values()):
            state = "correct_for_all_detectors"
        elif correct and not any(correct.values()):
            state = "wrong_for_all_detectors"
        elif sum(1 for v in correct.values() if v) == 1:
            only = next(name for name, ok in correct.items() if ok)
            state = f"correct_only_for_{only}"
        else:
            state = "detectors_disagree"
        states[image_id] = {
            "state": state,
            "detectors": values,
            "reduced_false_positives": _best_lower(fp),
            "reduced_false_negatives": _best_lower(fn),
        }
    return states


def _best_lower(values: Dict[str, int]) -> List[str]:
    if not values:
        return []
    best = min(values.values())
    return [name for name, value in values.items() if value == best]

