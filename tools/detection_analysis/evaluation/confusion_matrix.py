"""Detection-oriented confusion matrix."""

from __future__ import annotations

from typing import List, Tuple

import numpy as np
import pandas as pd

from tools.detection_analysis.constants import ERROR_CLASSIFICATION, ERROR_TRUE_POSITIVE, GT_FALSE_NEGATIVE, GT_POOR_LOCALIZATION
from tools.detection_analysis.models import GroundTruthMatchRecord, MatchRecord


def build_confusion_matrix(
    prediction_matches: List[MatchRecord],
    ground_truth_matches: List[GroundTruthMatchRecord],
    class_names: List[str],
    normalize: str = "raw",
) -> pd.DataFrame:
    """Build a matrix with GT classes as rows and predicted classes as columns."""

    labels = list(class_names) + ["background", "missed"]
    matrix = np.zeros((len(labels), len(labels)), dtype=float)
    background_idx = len(class_names)
    missed_idx = len(class_names) + 1
    for pred in prediction_matches:
        if pred.status == ERROR_TRUE_POSITIVE and pred.gt_class_id is not None:
            matrix[pred.gt_class_id, pred.pred_class_id] += 1
        elif pred.status == ERROR_CLASSIFICATION and pred.gt_class_id is not None:
            matrix[pred.gt_class_id, pred.pred_class_id] += 1
        elif pred.gt_class_id is None and 0 <= pred.pred_class_id < len(class_names):
            matrix[background_idx, pred.pred_class_id] += 1
    for gt in ground_truth_matches:
        if gt.status in {GT_FALSE_NEGATIVE, GT_POOR_LOCALIZATION}:
            matrix[gt.gt_class_id, missed_idx] += 1
    if normalize == "row":
        denom = matrix.sum(axis=1, keepdims=True)
        matrix = np.divide(matrix, denom, out=np.zeros_like(matrix), where=denom != 0)
    elif normalize == "column":
        denom = matrix.sum(axis=0, keepdims=True)
        matrix = np.divide(matrix, denom, out=np.zeros_like(matrix), where=denom != 0)
    return pd.DataFrame(matrix, index=labels, columns=labels)
