"""Precision-recall curve utilities."""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional

from tools.detection_analysis.constants import ERROR_TRUE_POSITIVE
from tools.detection_analysis.evaluation.custom_metrics import FP_STATUSES
from tools.detection_analysis.models import GroundTruthMatchRecord, MatchRecord


def precision_recall_curve(
    prediction_matches: Iterable[MatchRecord],
    ground_truth_matches: Iterable[GroundTruthMatchRecord],
    class_id: Optional[int] = None,
) -> List[Dict[str, float]]:
    """Return precision/recall/F1 as confidence threshold is swept downward."""

    preds = [
        p for p in prediction_matches
        if class_id is None or p.pred_class_id == class_id
    ]
    gts = [
        g for g in ground_truth_matches
        if class_id is None or g.gt_class_id == class_id
    ]
    total_gt = len([g for g in gts if "ignored" not in g.status and "crowd" not in g.status])
    tp = 0
    fp = 0
    rows: List[Dict[str, float]] = []
    for pred in sorted(preds, key=lambda p: p.score, reverse=True):
        if pred.status == ERROR_TRUE_POSITIVE:
            tp += 1
        elif pred.status in FP_STATUSES:
            fp += 1
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / total_gt if total_gt else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        rows.append({
            "confidence": pred.score,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        })
    return rows

